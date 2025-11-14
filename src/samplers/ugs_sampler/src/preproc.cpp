// preproc.cpp
// Preprocessing for UGS sampler: CSR, 1-DD ordering, bucket weights

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
#include <string>
#include <cstdlib>

#include "sampler.hpp"

namespace py = pybind11;

static int64_t next_handle = 1;

std::mutex registry_mutex;
std::unordered_map<i64, std::shared_ptr<Preproc>> registry;

//=============================================================================
// Build CSR adjacency list (undirected)
//=============================================================================
// Also tracks for each CSR entry which column in edge_index it came from

static void build_csr(
    const torch::Tensor& edge_index,
    i64 n,
    std::vector<i64>& indptr,
    std::vector<i32>& indices,
    std::vector<i32>& edge_col_of_csr_pos
) {
    TORCH_CHECK(edge_index.device().is_cpu(), "edge_index must be on CPU");
    TORCH_CHECK(edge_index.dtype() == torch::kInt64, "edge_index must be int64");

    auto ei = edge_index.accessor<i64, 2>();
    const i64 m = edge_index.size(1);

    // Count degrees (undirected: each edge contributes to both endpoints)
    indptr.assign(n + 1, 0);
    for (i64 j = 0; j < m; ++j) {
        const i64 u = ei[0][j];
        const i64 v = ei[1][j];
        if (u < 0 || v < 0 || u >= n || v >= n) continue;
        indptr[u + 1]++;
        indptr[v + 1]++;
    }

    // Prefix sum to get indptr
    for (i64 i = 1; i <= n; ++i) {
        indptr[i] += indptr[i - 1];
    }

    const i64 nnz = indptr[n];
    indices.assign((size_t)nnz, (i32)-1);
    edge_col_of_csr_pos.assign((size_t)nnz, (i32)-1);

    // Current write positions
    std::vector<i64> cur(n);
    for (i64 i = 0; i < n; ++i) {
        cur[i] = indptr[i];
    }

    // Fill CSR
    for (i64 j = 0; j < m; ++j) {
        const i64 u = ei[0][j];
        const i64 v = ei[1][j];
        if (u < 0 || v < 0 || u >= n || v >= n) continue;

        // u -> v
        i64 pos_u = cur[u]++;
        indices[pos_u] = (i32)v;
        edge_col_of_csr_pos[pos_u] = (i32)j;

        // v -> u (undirected)
        i64 pos_v = cur[v]++;
        indices[pos_v] = (i32)u;
        edge_col_of_csr_pos[pos_v] = (i32)j;
    }
}

//=============================================================================
// Compute 1-DD (degeneracy) ordering
//=============================================================================
// Algorithm: Repeatedly remove the highest-degree vertex
// Result: vertices ordered such that each vertex has at most d neighbors
// that come later in the ordering (d = degeneracy)
//
// This is the "reverse" of the standard degeneracy ordering used in the paper

static void compute_1dd_ordering(
    i64 n,
    const std::vector<i64>& indptr,
    const std::vector<i32>& indices,
    std::vector<int>& order_out,
    std::vector<int>& index_of_out
) {
    // Compute degrees
    std::vector<int> deg(n);
    int max_deg = 0;
    for (i64 v = 0; v < n; ++v) {
        int d = (int)(indptr[v + 1] - indptr[v]);
        deg[v] = d;
        if (d > max_deg) max_deg = d;
    }

    // Bucket sort by degree
    std::vector<std::vector<int>> buckets(max_deg + 1);
    for (int v = 0; v < (int)n; ++v) {
        buckets[deg[v]].push_back(v);
    }

    // Repeatedly remove max-degree vertex
    order_out.clear();
    order_out.reserve(n);
    std::vector<char> removed(n, 0);

    int cur_max = max_deg;
    for (i64 removed_count = 0; removed_count < n; ++removed_count) {
        // Find next non-empty bucket
        while (cur_max >= 0 && buckets[cur_max].empty()) {
            cur_max--;
        }
        assert(cur_max >= 0 && "All vertices removed");

        // Remove vertex from max bucket
        int v = buckets[cur_max].back();
        buckets[cur_max].pop_back();

        if (removed[v]) {
            removed_count--;
            continue;
        }

        removed[v] = 1;
        order_out.push_back(v);

        // Update neighbor degrees
        for (i64 p = indptr[v]; p < indptr[v + 1]; ++p) {
            int u = indices[p];
            if (removed[u]) continue;

            int old_deg = deg[u];
            deg[u] = old_deg - 1;

            if (old_deg - 1 >= 0) {
                buckets[old_deg - 1].push_back(u);
            }
        }
    }

    // Reverse to get ≺ ordering (smallest first)
    std::reverse(order_out.begin(), order_out.end());

    // Build inverse map
    index_of_out.assign(n, -1);
    for (size_t i = 0; i < order_out.size(); ++i) {
        index_of_out[order_out[i]] = (int)i;
    }
}

//=============================================================================
// Compute suffix degrees and bucket weights
//=============================================================================
// For each vertex position vi in the ordering:
// - suffix_deg[vi] = number of neighbors at position >= vi (in suffix graph)
// - bucket_b[vi] = (suffix_deg[vi])^(k-1) if BFS from vi reaches k vertices
//                = 0 otherwise

static void compute_suffix_and_buckets(std::shared_ptr<Preproc> P, int k) {
    const int n = (int)P->n;

    // Compute suffix degrees
    P->suffix_deg.assign(n, 0);
    for (int vi = 0; vi < n; ++vi) {
        int v = P->order[vi];
        int cnt = 0;
        for (i64 p = P->indptr[v]; p < P->indptr[v + 1]; ++p) {
            int u = P->indices[p];
            if (P->index_of[u] >= vi) {  // u is in suffix
                cnt++;
            }
        }
        P->suffix_deg[vi] = cnt;
    }

    // Compute bucket weights
    P->bucket_b.assign(n, 0.0);
    P->Z = 0.0;

    for (int vi = 0; vi < n; ++vi) {
        int v = P->order[vi];

        // Check if we can reach k vertices via BFS in suffix graph
        std::vector<int> bfs_queue;
        std::vector<char> visited(n, 0);

        bfs_queue.push_back(v);
        visited[v] = 1;

        for (size_t head = 0; head < bfs_queue.size() && (int)bfs_queue.size() < k; ++head) {
            int u = bfs_queue[head];

            for (i64 p = P->indptr[u]; p < P->indptr[u + 1]; ++p) {
                int w = P->indices[p];

                // Only consider suffix
                if (P->index_of[w] < vi) continue;

                // Only visit each vertex once
                if (visited[w]) continue;

                visited[w] = 1;
                bfs_queue.push_back(w);

                if ((int)bfs_queue.size() >= k) break;
            }
        }

        // If we can reach k vertices, compute bucket weight
        if ((int)bfs_queue.size() >= k) {
            int d_v = std::max(1, (int)P->suffix_deg[vi]);
            double b_v = 1.0;
            for (int t = 0; t < k - 1; ++t) {
                b_v *= (double)d_v;
            }
            P->bucket_b[vi] = b_v;
            P->Z += b_v;
        } else {
            P->bucket_b[vi] = 0.0;
        }
    }

    // Build alias table for weighted sampling
    if (P->Z > 0.0) {
        P->alias.build(P->bucket_b);
    }

    // Debug output
    static const char* debug_env = std::getenv("UGS_DEBUG");
    bool debug_mode = (debug_env && std::string(debug_env) == "1");
    if (debug_mode) {
        int viable = 0;
        for (int vi = 0; vi < n; ++vi) {
            if (P->bucket_b[vi] > 0.0) viable++;
        }
        std::fprintf(stderr, "[UGS PREPROC] n=%d k=%d Z=%.2e viable=%d/%d\n",
                     n, k, P->Z, viable, n);
    }
}

//=============================================================================
// Public API
//=============================================================================

int64_t create_preproc(const torch::Tensor& edge_index, i64 num_nodes, int k) {
    TORCH_CHECK(edge_index.device().is_cpu(), "edge_index must be on CPU");
    TORCH_CHECK(edge_index.dtype() == torch::kInt64, "edge_index must be int64");

    auto P = std::make_shared<Preproc>();
    P->n = num_nodes;

    // Build CSR
    build_csr(edge_index, P->n, P->indptr, P->indices, P->edge_col_of_csr_pos);
    P->m = (i64)P->indices.size();

    // Compute 1-DD ordering
    compute_1dd_ordering(P->n, P->indptr, P->indices, P->order, P->index_of);

    // Compute suffix degrees and bucket weights
    compute_suffix_and_buckets(P, k);

    // Register handle
    std::lock_guard<std::mutex> lock(registry_mutex);
    int64_t handle = next_handle++;
    registry[handle] = std::move(P);
    return handle;
}

void destroy_preproc(int64_t handle) {
    std::lock_guard<std::mutex> lock(registry_mutex);
    auto it = registry.find(handle);
    if (it != registry.end()) {
        registry.erase(it);
    }
}

bool has_graphlets(int64_t handle) {
    std::lock_guard<std::mutex> lock(registry_mutex);
    auto it = registry.find(handle);
    if (it == registry.end()) return false;
    return it->second->Z > 0.0;
}

py::dict get_preproc_info(int64_t handle) {
    py::dict d;
    std::lock_guard<std::mutex> lock(registry_mutex);
    auto it = registry.find(handle);
    if (it == registry.end()) return d;

    auto& P = it->second;
    d["num_nodes"] = (long long)P->n;
    d["num_edges_stored"] = (long long)P->m;
    d["Z"] = P->Z;
    d["bucket_count_nonzero"] = (int)std::count_if(
        P->bucket_b.begin(), P->bucket_b.end(), [](double x) { return x > 0.0; });
    return d;
}
