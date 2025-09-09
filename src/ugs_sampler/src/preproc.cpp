// preproc.cpp
// Preprocessing for Ugs sampler: CSR, 1-DD order, suffix degrees, bucket weights, alias table.
// Build with: -std=c++17 -O3 and link to pybind11 + libtorch if desired.

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <cstdint>

#include "sampler.hpp"

namespace py = pybind11;

static int64_t next_handle = 1;

std::mutex registry_mutex;
std::unordered_map<i64, std::shared_ptr<Preproc>> registry;

// ----------Private Functions----------

// Build CSR (undirected) from edge_index [2, m] tensor (CPU, int64)
static void build_csr(const torch::Tensor &edge_index, i64 n,
                      std::vector<i64> &indptr, std::vector<i32> &indices) {
    TORCH_CHECK(edge_index.device().is_cpu(), "edge_index must be on CPU");
    TORCH_CHECK(edge_index.dtype() == torch::kInt64, "edge_index must be int64");
    auto ei = edge_index.accessor<i64,2>();
    i64 m = edge_index.size(1);
    indptr.assign(n+1, 0);
    for (i64 j=0;j<m;++j) {
        i64 u = ei[0][j];
        i64 v = ei[1][j];
        if (u<0 || v<0 || u>=n || v>=n) continue;
        indptr[u+1]++; indptr[v+1]++;
    }
    for (i64 i=1;i<=n;++i) indptr[i] += indptr[i-1];
    indices.assign((size_t)indptr[n], -1);
    std::vector<i64> cur(n);
    for (i64 i=0;i<n;++i) cur[i] = indptr[i];
    for (i64 j=0;j<m;++j) {
        i64 u = ei[0][j];
        i64 v = ei[1][j];
        if (u<0 || v<0 || u>=n || v>=n) continue;
        indices[(size_t)cur[u]++] = (i32)v;
        indices[(size_t)cur[v]++] = (i32)u;
    }
}

// Compute 1-DD order by repeatedly removing max-degree vertex using degree buckets (O(n+m) amortized)
static void compute_1dd_order(i64 n,
                              const std::vector<i64> &indptr,
                              const std::vector<i32> &indices,
                              std::vector<int> &order_out,
                              std::vector<int> &index_of_out) {
    std::vector<int> deg((size_t)n);
    int maxdeg = 0;
    for (i64 v=0; v<n; ++v) {
        int d = (int)(indptr[v+1] - indptr[v]);
        deg[(size_t)v] = d;
        if (d > maxdeg) maxdeg = d;
    }
    std::vector<std::vector<int>> buckets(maxdeg + 1);
    for (int v = 0; v < (int)n; ++v) buckets[deg[v]].push_back(v);
    order_out.clear(); order_out.reserve((size_t)n);
    std::vector<char> removed((size_t)n, 0);
    int cur_max = maxdeg;
    for (i64 removed_count = 0; removed_count < n; ++removed_count) {
        while (cur_max >= 0 && buckets[cur_max].empty()) --cur_max;
        assert(cur_max >= 0);
        int v = buckets[cur_max].back();
        buckets[cur_max].pop_back();
        if (removed[v]) { --removed_count; continue; }
        removed[v] = 1;
        order_out.push_back(v);
        // update neighbor degrees
        for (i64 p = indptr[v]; p < indptr[v+1]; ++p) {
            int u = indices[(size_t)p];
            if (removed[u]) continue;
            int du = deg[u];
            deg[u] = du - 1;
            if (du - 1 >= 0) buckets[du-1].push_back(u);
        }
    }
    // reverse to obtain ≺ (smallest first)
    std::reverse(order_out.begin(), order_out.end());
    index_of_out.assign((size_t)n, -1);
    for (size_t i = 0; i < order_out.size(); ++i) index_of_out[order_out[i]] = (int)i;
}

// Compute suffix degrees d(v | G(v)) and bucket weights b[vi] (vi indexes order positions)
static void compute_suffix_and_buckets(std::shared_ptr<Preproc> P, int k) {
    int n = (int)P->n;
    P->suffix_deg.assign((size_t)n, 0);
    // suffix_deg[vi] = number of neighbors of vertex order[vi] that are in positions >= vi
    for (int vi = 0; vi < n; ++vi) {
        int v = P->order[vi];
        int cnt = 0;
        for (i64 p = P->indptr[v]; p < P->indptr[v+1]; ++p) {
            int u = P->indices[(size_t)p];
            if (P->index_of[u] >= vi) ++cnt;
        }
        P->suffix_deg[vi] = (i32)cnt;
    }
    // detect if bucket non-empty (B(v) != ∅) by truncated BFS up to k nodes; for small k this is cheap
    P->bucket_b.assign((size_t)n, 0.0);
    P->Z = 0.0;
    std::vector<int> bfsq; bfsq.reserve(k*2);
    for (int vi=0; vi<n; ++vi) {
        int v = P->order[vi];
        // quick negative check
        if (P->suffix_deg[vi] == 0 && k > 1) { P->bucket_b[vi] = 0.0; continue; }
        // truncated BFS within suffix nodes
        bfsq.clear();
        bfsq.push_back(v);
        size_t idx = 0;
        // small local seen set implemented as linear scan on bfsq (k small)
        while (idx < bfsq.size() && (int)bfsq.size() < k) {
            int x = bfsq[idx++];
            for (i64 p = P->indptr[x]; p < P->indptr[x+1] && (int)bfsq.size() < k; ++p) {
                int y = P->indices[(size_t)p];
                if (P->index_of[y] < vi) continue; // outside suffix
                bool present = false;
                for (int z : bfsq) if (z == y) { present = true; break; }
                if (!present) bfsq.push_back(y);
            }
        }
        if ((int)bfsq.size() >= k) {
            // bucket non-empty; weight ~ max(1, suffix_deg^(k-1))
            int dv = std::max(1, (int)P->suffix_deg[vi]);
            double bv = 1.0;
            for (int t=0;t<k-1;++t) bv *= (double)dv;
            P->bucket_b[vi] = bv;
            P->Z += bv;
        } else {
            P->bucket_b[vi] = 0.0;
        }
    }
    // build alias distribution over all positions (weights may include zeros)
    if (P->Z > 0.0) {
        P->alias.build(P->bucket_b);
    }
}

// ---------- Public API exposed to Python (pybind11) ----------

// create_preproc(edge_index: LongTensor [2, m], num_nodes: int, k: int) -> int64 handle
int64_t create_preproc(const torch::Tensor &edge_index, i64 num_nodes, int k) {
    TORCH_CHECK(edge_index.device().is_cpu(), "edge_index must be on CPU");
    TORCH_CHECK(edge_index.dtype() == torch::kInt64, "edge_index must be int64");
    auto P = std::make_shared<Preproc>();
    P->n = num_nodes;
    // CSR
    build_csr(edge_index, P->n, P->indptr, P->indices);
    P->m = (i64)P->indices.size();
    // 1-DD order
    compute_1dd_order(P->n, P->indptr, P->indices, P->order, P->index_of);
    // suffix degrees & bucket weights
    compute_suffix_and_buckets(P, k);
    // register
    std::lock_guard<std::mutex> lock(registry_mutex);
    int64_t handle = next_handle++;
    registry[handle] = std::move(P);
    return handle;
}

// destroy handle
void destroy_preproc(int64_t handle) {
    std::lock_guard<std::mutex> lock(registry_mutex);
    auto it = registry.find(handle);
    if (it != registry.end()) registry.erase(it);
}

// helper to check if this preproc has any graphlets (Z>0)
bool has_graphlets(int64_t handle) {
    std::lock_guard<std::mutex> lock(registry_mutex);
    auto it = registry.find(handle);
    if (it == registry.end()) return false;
    return it->second->Z > 0.0;
}

// expose a way to fetch some small metadata (for debugging) -- optional
py::dict get_preproc_info(int64_t handle) {
    py::dict d;
    std::lock_guard<std::mutex> lock(registry_mutex);
    auto it = registry.find(handle);
    if (it == registry.end()) return d;
    auto &P = it->second;
    d["num_nodes"] = (long long)P->n;
    d["num_edges_stored"] = (long long)P->m;
    d["Z"] = P->Z;
    d["bucket_count_nonzero"] = (int)std::count_if(P->bucket_b.begin(), P->bucket_b.end(), [](double x){ return x > 0.0; });
    return d;
}
