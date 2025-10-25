// sampler.cpp
// UGS sampling implementation — relies on preprocessing from preproc.cpp

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <vector>
#include <random>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <chrono>
#include <algorithm> // for std::find

#include "sampler.hpp"

namespace py = pybind11;
using i64 = int64_t;

// --- Rand-Grow helper (trimmed; unchanged logic) ---
static void rand_grow_sample(const Preproc &P, int k, ThreadRNG &rng,
                             std::vector<int> &out_seq, int root_vi) {
    int root = P.order[root_vi];
    out_seq.clear();
    out_seq.reserve(k);
    out_seq.push_back(root);

    for (int step = 1; step < k; ++step) {
        std::vector<int> cut;
        for (int u : out_seq) {
            for (i64 p = P.indptr[u]; p < P.indptr[u + 1]; ++p) {
                int w = P.indices[p];
                if (P.index_of[w] < root_vi) continue;
                if (std::find(out_seq.begin(), out_seq.end(), w) == out_seq.end())
                    cut.push_back(w);
            }
        }
        if (cut.empty()) return;
        int newv = cut[rng.next_int((int)cut.size())];
        out_seq.push_back(newv);
    }
}

// --- Sample API ---
//
// Returns:
// 0) nodes_t          [B, k]        (graph-local ids; caller may add base_offset if needed)
// 1) edge_index_t     [2, E_emit]   (endpoints per edge_mode; may be trimmed from prealloc)
// 2) edge_ptr_t       [B+1]         (CSR over samples)
// 3) edge_src_idx_t   [E_emit]      (per-graph LOCAL edge column ids, aligned with edge_index_t)
//
py::tuple sample(i64 handle, int m_per_graph, int k,
                 std::string edge_mode = "local",
                 int64_t base_offset = 0) {
    std::shared_ptr<Preproc> P;
    {
        std::lock_guard<std::mutex> lock(registry_mutex);
        auto it = registry.find(handle);
        if (it == registry.end()) throw std::runtime_error("Invalid preproc handle");
        P = it->second;
    }

    const int B = m_per_graph;
    std::vector<std::vector<int>> samples(B);

    std::vector<int> viable_roots;
    viable_roots.reserve(P->order.size());
    for (int vi = 0; vi < (int)P->order.size(); ++vi)
        if (P->bucket_b[vi] > 0.0) viable_roots.push_back(vi);

    // Relaxations
    if (viable_roots.empty()) {
        for (int vi = 0; vi < (int)P->order.size(); ++vi)
            if (P->suffix_deg[vi] > 0) viable_roots.push_back(vi);
    }
    if (viable_roots.empty()) {
        for (int vi = 0; vi < (int)P->order.size(); ++vi)
            viable_roots.push_back(vi);
    }
    if (viable_roots.empty()) {
        std::fprintf(stderr, "sample: no viable_roots at all (n_pos=%zu, k=%d)\n",
                     P->order.size(), k);
        throw std::runtime_error("No viable roots available");
    }

    // Draw node sets
    for (int b = 0; b < B; ++b) {
        if (viable_roots.empty()) throw std::runtime_error("No viable roots available");
        uint64_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count() ^ (uint64_t)b;
        ThreadRNG rng(seed);
        int root_vi, tries = 0;
        do {
            root_vi = viable_roots[rng.next_int((int)viable_roots.size())];
            rand_grow_sample(*P, k, rng, samples[b], root_vi);
            ++tries;
        } while ((int)samples[b].size() < k && tries < 100);
    }

    // Gather candidate edges per sample, keeping CSR position 'p' for each edge
    struct EdgeTrip { int u_local; int v_local; i64 csr_pos; };
    std::vector<std::vector<EdgeTrip>> samples_edges(B);
    i64 total_edges = 0;

    for (int b = 0; b < B; ++b) {
        const auto &nodes = samples[b];
        if ((int)nodes.size() < k) continue; // failed sample: no edges

        // global->local map within this sample's k-set
        std::unordered_map<int,int> g2l;
        g2l.reserve((size_t)k * 2);
        for (int i = 0; i < k; ++i) g2l[nodes[i]] = i;

        auto &edges = samples_edges[b];
        for (int i = 0; i < k; ++i) {
            int u = nodes[i];
            for (i64 p = P->indptr[u]; p < P->indptr[u + 1]; ++p) {
                int v = P->indices[p];
                auto it = g2l.find(v);
                if (it != g2l.end()) {
                    edges.push_back({ i, it->second, p }); // keep CSR linear index 'p'
                }
            }
        }
        total_edges += (i64)edges.size();
    }

    auto opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU).pinned_memory(true);

    torch::Tensor nodes_t          = torch::full({B, k}, -1, opts);
    torch::Tensor edge_ptr_t       = torch::empty({B + 1}, opts);
    torch::Tensor edge_index_t     = torch::empty({2, total_edges}, opts);
    torch::Tensor edge_src_idx_t   = torch::empty({total_edges}, opts);  // NEW

    auto nodes_ptr     = nodes_t.data_ptr<i64>();
    auto edge_ptr_ptr  = edge_ptr_t.data_ptr<i64>();
    auto edge_idx_ptr  = edge_index_t.data_ptr<i64>();
    auto edge_src_ptr  = edge_src_idx_t.data_ptr<i64>();

    edge_ptr_ptr[0] = 0;
    i64 epos = 0;

    for (int b = 0; b < B; ++b) {
        // write nodes (graph-local ids; caller can globalize if desired)
        for (int i = 0; i < (int)samples[b].size() && i < k; ++i)
            nodes_ptr[b * (i64)k + i] = samples[b][i];
        for (int i = (int)samples[b].size(); i < k; ++i)
            nodes_ptr[b * (i64)k + i] = -1;

        // write edges (+ src local ids), respecting edge_mode
        for (const auto &e : samples_edges[b]) {
            const int u_local = e.u_local;
            const int v_local = e.v_local;

            // read graph-local node ids we just wrote
            const i64 u_gl = nodes_ptr[b * (i64)k + (i64)u_local];
            const i64 v_gl = nodes_ptr[b * (i64)k + (i64)v_local];
            if (u_gl == -1 || v_gl == -1) continue; // guard (degenerate sample)

            i64 u_out = 0, v_out = 0;
            if (edge_mode == "local") {
                u_out = (i64)u_local;
                v_out = (i64)v_local;
            } else if (edge_mode == "flat") {
                u_out = (i64)b * (i64)k + (i64)u_local;
                v_out = (i64)b * (i64)k + (i64)v_local;
            } else if (edge_mode == "global") {
                // graph-local -> dataset-global via base_offset
                u_out = u_gl + base_offset;
                v_out = v_gl + base_offset;
            } else {
                throw std::runtime_error(std::string("Unknown edge_mode: ") + edge_mode);
            }

            // write edge
            edge_idx_ptr[epos]                 = u_out;
            edge_idx_ptr[total_edges + epos]   = v_out;

            edge_src_ptr[epos] = (i64)P->edge_col_of_csr_pos[(size_t)e.csr_pos];  // 0..E_graph-1

            ++epos;
        }

        edge_ptr_ptr[b + 1] = epos;
    }

    // Trim if we skipped any edges
    if (epos < total_edges) {
        // edge_index trim
        torch::Tensor edge_index_trim = torch::empty({2, epos}, opts);
        auto trim_ptr = edge_index_trim.data_ptr<i64>();
        for (i64 i = 0; i < epos; ++i) {
            trim_ptr[i]       = edge_idx_ptr[i];
            trim_ptr[epos + i] = edge_idx_ptr[total_edges + i];
        }
        edge_index_t = edge_index_trim;

        // edge_src_idx trim
        torch::Tensor edge_src_trim = torch::empty({epos}, opts);
        std::memcpy(edge_src_trim.data_ptr<i64>(), edge_src_ptr, sizeof(i64) * (size_t)epos);
        edge_src_idx_t = edge_src_trim;
    }

    // Return minimal pack for downstream:
    // nodes_t, edge_index_t, edge_ptr_t, edge_src_idx_t
    return py::make_tuple(nodes_t, edge_index_t, edge_ptr_t, edge_src_idx_t);
}