// sampler.cpp
// Clean implementation of UGS (Uniform connected Graphlet Sampling)
// Based on: "Efficient and near-optimal algorithms for sampling small connected subgraphs"

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <chrono>
#include <random>
#include <string>
#include <cstdlib>

#include "sampler.hpp"

namespace py = pybind11;
using i64 = int64_t;

//=============================================================================
// Rand-Grow: Uniformly grow a connected k-subgraph from a root
//=============================================================================
// Algorithm:
// 1. Start with root vertex v (from ordering position root_vi)
// 2. At each step:
//    - Compute "cut" = neighbors of current subgraph NOT in subgraph
//    - Only consider vertices in suffix: index_of[w] >= root_vi
//    - Uniformly select one vertex from cut and add to subgraph
// 3. Return when subgraph has k vertices
//
// KEY: Each vertex appears in cut at most ONCE (use set for deduplication)
//=============================================================================

static void rand_grow(
    const Preproc& P,
    int k,
    int root_vi,
    ThreadRNG& rng,
    std::vector<int>& out_vertices  // Output: vertex IDs (not order positions)
) {
    int root = P.order[root_vi];
    out_vertices.clear();
    out_vertices.reserve(k);
    out_vertices.push_back(root);

    // Track which vertices are in the subgraph (for fast membership test)
    std::unordered_set<int> in_subgraph;
    in_subgraph.insert(root);

    // Grow k-1 more vertices
    for (int step = 1; step < k; ++step) {
        // Build cut: neighbors of subgraph not yet in subgraph
        std::unordered_set<int> cut_set;

        for (int u : out_vertices) {
            // Iterate over neighbors of u
            for (i64 p = P.indptr[u]; p < P.indptr[u + 1]; ++p) {
                int w = P.indices[p];

                // Only consider vertices in suffix (at or after root in ordering)
                if (P.index_of[w] < root_vi) continue;

                // Only consider vertices not yet in subgraph
                if (in_subgraph.count(w)) continue;

                // Add to cut (set ensures no duplicates)
                cut_set.insert(w);
            }
        }

        // If no neighbors available, growth failed
        if (cut_set.empty()) {
            return;  // Return partial subgraph
        }

        // Uniformly select one vertex from cut
        std::vector<int> cut(cut_set.begin(), cut_set.end());
        int next = cut[rng.next_int((int)cut.size())];

        out_vertices.push_back(next);
        in_subgraph.insert(next);
    }
}

//=============================================================================
// Sample: Generate m subgraphs of size k from preprocessed graph
//=============================================================================

py::tuple sample(
    i64 handle,
    int m,
    int k,
    std::string edge_mode,  // "local" | "flat" | "global"
    int64_t base_offset,
    int seed
) {
    // Get preprocessing data
    std::shared_ptr<Preproc> P;
    {
        std::lock_guard<std::mutex> lock(registry_mutex);
        auto it = registry.find(handle);
        if (it == registry.end()) {
            throw std::runtime_error("Invalid preproc handle");
        }
        P = it->second;
    }

    // Check for debug mode
    static const char* debug_env = std::getenv("UGS_DEBUG");
    static const bool debug_mode = (debug_env && std::string(debug_env) == "1");

    const int n = (int)P->n;

    // Find viable roots (those with b[vi] > 0)
    std::vector<int> viable_roots;
    for (int vi = 0; vi < n; ++vi) {
        if (P->bucket_b[vi] > 0.0) {
            viable_roots.push_back(vi);
        }
    }

    // Fallback if no viable roots
    int relaxation_level = 0;
    if (viable_roots.empty()) {
        relaxation_level = 1;
        if (debug_mode) {
            std::fprintf(stderr, "[UGS WARNING] No roots with b>0 (Z=%.2e). Using suffix_deg>0 (BREAKS UNIFORMITY)\n", P->Z);
        }
        for (int vi = 0; vi < n; ++vi) {
            if (P->suffix_deg[vi] > 0) {
                viable_roots.push_back(vi);
            }
        }
    }

    if (viable_roots.empty()) {
        relaxation_level = 2;
        if (debug_mode) {
            std::fprintf(stderr, "[UGS WARNING] No roots with suffix_deg>0. Using all nodes (BREAKS UNIFORMITY)\n");
        }
        for (int vi = 0; vi < n; ++vi) {
            viable_roots.push_back(vi);
        }
    }

    if (viable_roots.empty()) {
        throw std::runtime_error("No viable roots available");
    }

    // Sample m subgraphs
    std::vector<std::vector<int>> samples(m);
    int incomplete = 0;
    int weighted_count = 0;
    int uniform_count = 0;

    for (int i = 0; i < m; ++i) {
        // Create RNG for this sample
        uint64_t rng_seed = (uint64_t)seed + (uint64_t)i * 0x9e3779b97f4a7c15ULL;  // Mix in sample index
        ThreadRNG rng(rng_seed);

        // Select root
        int root_vi;
        if (relaxation_level == 0 && P->Z > 0.0) {
            // Weighted selection via alias table
            root_vi = P->alias.sample(rng);
            weighted_count++;
        } else {
            // Uniform selection (fallback)
            root_vi = viable_roots[rng.next_int((int)viable_roots.size())];
            uniform_count++;
        }

        // Grow subgraph
        rand_grow(*P, k, root_vi, rng, samples[i]);

        if ((int)samples[i].size() < k) {
            incomplete++;
        }
    }

    // Report diagnostics
    if (debug_mode) {
        std::fprintf(stderr, "[UGS SAMPLE] n=%d k=%d m=%d Z=%.2e | relax=%d weighted=%d uniform=%d incomplete=%d\n",
                     n, k, m, P->Z, relaxation_level, weighted_count, uniform_count, incomplete);
    }

    // Build output tensors
    auto opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU).pinned_memory(true);

    // Nodes tensor [m, k]
    torch::Tensor nodes_t = torch::full({m, k}, -1, opts);
    auto nodes_acc = nodes_t.accessor<i64, 2>();

    // Edge structures
    std::vector<i64> edge_ptr_vec;
    edge_ptr_vec.reserve(m + 1);
    edge_ptr_vec.push_back(0);

    struct EdgeData { int u_local; int v_local; i64 csr_pos; };
    std::vector<std::vector<EdgeData>> sample_edges(m);

    // Process each sample
    for (int i = 0; i < m; ++i) {
        const auto& verts = samples[i];
        const int size = (int)verts.size();

        // Write node IDs
        for (int j = 0; j < size && j < k; ++j) {
            i64 node_id = verts[j];
            // Apply coordinate transformation based on edge_mode
            if (edge_mode == "global") {
                node_id += base_offset;
            }
            nodes_acc[i][j] = node_id;
        }

        // Skip edges if sample is incomplete
        if (size < k) {
            edge_ptr_vec.push_back(edge_ptr_vec.back());
            continue;
        }

        // Build local node index map
        std::unordered_map<int, int> global_to_local;
        for (int j = 0; j < size; ++j) {
            global_to_local[verts[j]] = j;
        }

        // Extract edges within subgraph
        for (int j = 0; j < size; ++j) {
            int u = verts[j];
            for (i64 p = P->indptr[u]; p < P->indptr[u + 1]; ++p) {
                int v = P->indices[p];
                auto it = global_to_local.find(v);
                if (it != global_to_local.end()) {
                    int v_local = it->second;
                    // Store edge (u_local, v_local) and its CSR position
                    sample_edges[i].push_back({j, v_local, p});
                }
            }
        }

        edge_ptr_vec.push_back(edge_ptr_vec.back() + (i64)sample_edges[i].size());
    }

    // Build edge tensors
    i64 total_edges = edge_ptr_vec.back();
    torch::Tensor edge_index_t = torch::empty({2, total_edges}, opts);
    torch::Tensor edge_src_t = torch::empty({total_edges}, opts);

    auto edge_u = edge_index_t.data_ptr<i64>();
    auto edge_v = edge_u + total_edges;
    auto edge_src = edge_src_t.data_ptr<i64>();

    i64 pos = 0;
    for (int i = 0; i < m; ++i) {
        for (const auto& e : sample_edges[i]) {
            // Compute final edge endpoints based on mode
            i64 u_final, v_final;
            if (edge_mode == "local") {
                u_final = e.u_local;
                v_final = e.v_local;
            } else if (edge_mode == "flat") {
                u_final = (i64)i * k + e.u_local;
                v_final = (i64)i * k + e.v_local;
            } else {  // global
                i64 u_global = nodes_acc[i][e.u_local];
                i64 v_global = nodes_acc[i][e.v_local];
                u_final = u_global;
                v_final = v_global;
            }

            edge_u[pos] = u_final;
            edge_v[pos] = v_final;
            edge_src[pos] = P->edge_col_of_csr_pos[e.csr_pos];
            pos++;
        }
    }

    // Build edge_ptr tensor
    torch::Tensor edge_ptr_t = torch::empty({(i64)edge_ptr_vec.size()}, opts);
    auto edge_ptr_acc = edge_ptr_t.accessor<i64, 1>();
    for (size_t i = 0; i < edge_ptr_vec.size(); ++i) {
        edge_ptr_acc[i] = edge_ptr_vec[i];
    }

    return py::make_tuple(nodes_t, edge_index_t, edge_ptr_t, edge_src_t);
}
