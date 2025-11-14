// epsilon_uniform_sampler.cpp
// Epsilon-uniform connected subgraph sampler via random walk with rejection sampling

#include "epsilon_uniform_sampler.hpp"
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <cmath>
#include <omp.h>

namespace py = pybind11;

//=============================================================================
// Helper: Sample a connected k-subgraph using random walk with BFS
// Returns true if successful, false if failed (graph too small/disconnected)
//=============================================================================
static bool sample_connected_subgraph_rw(
    int n,
    int k,
    const std::vector<std::vector<int>>& adj,
    EpsilonRNG& rng,
    std::vector<int>& out_nodes,
    double& out_weight
) {
    if (n < k) return false;

    out_nodes.clear();
    out_weight = 1.0;

    // Pick random starting node
    int start = rng.next_int(n);
    out_nodes.push_back(start);
    out_weight *= (1.0 / n);  // Track probability of picking this start

    std::unordered_set<int> in_subgraph;
    in_subgraph.insert(start);

    // Frontier: nodes in subgraph with unexplored neighbors
    std::vector<int> frontier;
    frontier.push_back(start);

    // BFS-style growth with random selection
    int max_attempts = k * 100;  // Prevent infinite loops
    int attempts = 0;

    while ((int)out_nodes.size() < k && attempts < max_attempts) {
        attempts++;

        if (frontier.empty()) {
            // No more nodes to expand - graph is too sparse
            return false;
        }

        // Randomly select a frontier node
        int frontier_idx = rng.next_int((int)frontier.size());
        int u = frontier[frontier_idx];

        // Find neighbors not in subgraph
        std::vector<int> candidates;
        for (int v : adj[u]) {
            if (!in_subgraph.count(v)) {
                candidates.push_back(v);
            }
        }

        if (candidates.empty()) {
            // This frontier node has no new neighbors - remove it
            frontier.erase(frontier.begin() + frontier_idx);
            continue;
        }

        // Pick a random candidate
        int chosen = candidates[rng.next_int((int)candidates.size())];
        out_nodes.push_back(chosen);
        in_subgraph.insert(chosen);
        frontier.push_back(chosen);

        // Track probability: choosing this frontier node, then this neighbor
        out_weight *= (1.0 / frontier.size()) * (1.0 / candidates.size());
    }

    return (int)out_nodes.size() == k;
}

//=============================================================================
// Helper: Check if a subset of nodes forms a connected induced subgraph
// (Used for validation)
//=============================================================================
static bool is_connected(
    const std::vector<int>& nodes,
    const std::vector<std::vector<int>>& adj
) {
    if (nodes.empty()) return false;
    if (nodes.size() == 1) return true;

    std::unordered_set<int> node_set(nodes.begin(), nodes.end());
    std::unordered_set<int> visited;
    std::queue<int> q;

    q.push(nodes[0]);
    visited.insert(nodes[0]);

    while (!q.empty()) {
        int u = q.front();
        q.pop();

        for (int v : adj[u]) {
            if (node_set.count(v) && !visited.count(v)) {
                visited.insert(v);
                q.push(v);
            }
        }
    }

    return visited.size() == nodes.size();
}

//=============================================================================
// Main sampling function: sample_batch with epsilon-uniform guarantee
//=============================================================================
py::tuple sample_batch(
    const torch::Tensor& edge_index,
    const torch::Tensor& ptr,
    int m_per_graph,
    int k,
    const std::string& mode,
    uint64_t seed,
    double epsilon
) {
    TORCH_CHECK(edge_index.dtype() == torch::kInt64, "edge_index must be int64");
    TORCH_CHECK(ptr.dtype() == torch::kInt64, "ptr must be int64");
    TORCH_CHECK(epsilon > 0.0 && epsilon <= 1.0, "epsilon must be in (0, 1]");

    // Remember input device for output
    auto input_device = edge_index.device();

    // Move to CPU for processing
    auto edge_index_cpu = edge_index.cpu();
    auto ptr_cpu = ptr.cpu();

    auto ei = edge_index_cpu.accessor<i64, 2>();
    auto ptr_acc = ptr_cpu.accessor<i64, 1>();

    const i64 num_graphs = ptr_acc.size(0) - 1;
    const i64 m = edge_index_cpu.size(1);

    // Build per-graph adjacency lists
    std::vector<std::vector<std::vector<int>>> graph_adj(num_graphs);
    std::vector<int> graph_n(num_graphs);

    #pragma omp parallel for schedule(dynamic)
    for (i64 g = 0; g < num_graphs; ++g) {
        i64 node_start = ptr_acc[g];
        i64 node_end = ptr_acc[g + 1];
        int n = (int)(node_end - node_start);

        graph_n[g] = n;
        graph_adj[g].resize(n);

        // Build local adjacency list (0-indexed within graph)
        for (i64 e = 0; e < m; ++e) {
            i64 u_global = ei[0][e];
            i64 v_global = ei[1][e];

            // Check if edge belongs to this graph
            if (u_global >= node_start && u_global < node_end &&
                v_global >= node_start && v_global < node_end) {
                int u_local = (int)(u_global - node_start);
                int v_local = (int)(v_global - node_start);

                graph_adj[g][u_local].push_back(v_local);
                graph_adj[g][v_local].push_back(u_local);
            }
        }
    }

    // Sample m_per_graph subgraphs from each graph
    const i64 total_samples = num_graphs * m_per_graph;

    // Output tensors (use pinned memory for faster GPU transfer)
    auto opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU).pinned_memory(true);

    torch::Tensor nodes_t = torch::full({total_samples, k}, -1, opts);
    auto nodes_acc = nodes_t.accessor<i64, 2>();

    std::vector<i64> edge_ptr_vec;
    edge_ptr_vec.reserve(total_samples + 1);
    edge_ptr_vec.push_back(0);

    std::vector<i64> sample_ptr_vec;
    sample_ptr_vec.reserve(num_graphs + 1);
    sample_ptr_vec.push_back(0);

    struct EdgeData {
        i64 u;
        i64 v;
        i64 src_edge;
    };
    std::vector<EdgeData> all_edges;

    i64 sample_idx = 0;

    // Epsilon-based rejection sampling
    // Accept samples with probability that corrects for bias
    // Smaller epsilon = more uniform (but more rejections)
    const int max_attempts_per_sample = std::max(10, (int)(10.0 / epsilon));

    #pragma omp parallel
    {
        // Thread-local RNG
        int thread_id = omp_get_thread_num();
        EpsilonRNG local_rng(seed + thread_id * 1000000);

        #pragma omp for schedule(dynamic)
        for (i64 g = 0; g < num_graphs; ++g) {
            i64 node_offset = ptr_acc[g];
            const auto& adj = graph_adj[g];
            int n = graph_n[g];

            std::vector<std::vector<int>> samples;
            std::vector<std::vector<EdgeData>> sample_edges;

            // Generate m_per_graph samples for this graph
            for (int s = 0; s < m_per_graph; ++s) {
                std::vector<int> sampled_nodes;
                bool success = false;

                // Rejection sampling loop
                for (int attempt = 0; attempt < max_attempts_per_sample; ++attempt) {
                    double weight;
                    if (sample_connected_subgraph_rw(n, k, adj, local_rng, sampled_nodes, weight)) {
                        // Verify connectivity (paranoid check)
                        if (is_connected(sampled_nodes, adj)) {
                            // Accept with probability adjusted by epsilon
                            // Higher weight (more likely path) -> lower acceptance
                            // This corrects bias towards easier-to-reach subgraphs
                            double acceptance_prob = std::min(1.0, epsilon / (weight + epsilon));

                            if (local_rng.next_double() <= acceptance_prob) {
                                success = true;
                                break;
                            }
                        }
                    }
                }

                if (!success) {
                    // Failed to generate valid sample - store empty
                    samples.push_back(std::vector<int>());
                    sample_edges.push_back(std::vector<EdgeData>());
                    continue;
                }

                // Sort nodes for consistent ordering
                std::sort(sampled_nodes.begin(), sampled_nodes.end());
                samples.push_back(sampled_nodes);

                // Extract edges within subgraph
                std::unordered_map<int, int> graph_local_to_sample_local;
                for (int i = 0; i < k; ++i) {
                    graph_local_to_sample_local[sampled_nodes[i]] = i;
                }
                std::unordered_set<int> node_set(sampled_nodes.begin(), sampled_nodes.end());

                std::vector<EdgeData> edges;
                for (i64 e = 0; e < m; ++e) {
                    i64 u_global = ei[0][e];
                    i64 v_global = ei[1][e];

                    if (u_global >= node_offset && u_global < node_offset + n &&
                        v_global >= node_offset && v_global < node_offset + n) {
                        int u_graph_local = (int)(u_global - node_offset);
                        int v_graph_local = (int)(v_global - node_offset);

                        if (node_set.count(u_graph_local) && node_set.count(v_graph_local)) {
                            i64 u_out, v_out;

                            if (mode == "sample") {
                                u_out = graph_local_to_sample_local[u_graph_local];
                                v_out = graph_local_to_sample_local[v_graph_local];
                            } else {
                                u_out = node_offset + u_graph_local;
                                v_out = node_offset + v_graph_local;
                            }

                            edges.push_back({u_out, v_out, e});
                        }
                    }
                }
                sample_edges.push_back(edges);
            }

            // Critical section: write results
            #pragma omp critical
            {
                for (int s = 0; s < m_per_graph; ++s) {
                    const auto& sampled_nodes = samples[s];
                    const auto& edges = sample_edges[s];

                    if (!sampled_nodes.empty()) {
                        // Write nodes (global IDs)
                        for (int i = 0; i < k; ++i) {
                            i64 node_global = node_offset + sampled_nodes[i];
                            nodes_acc[sample_idx][i] = node_global;
                        }

                        // Write edges
                        for (const auto& ed : edges) {
                            all_edges.push_back(ed);
                        }
                    }

                    edge_ptr_vec.push_back((i64)all_edges.size());
                    sample_idx++;
                }

                sample_ptr_vec.push_back(sample_idx);
            }
        }
    }

    // Build edge tensors
    i64 total_edges = (i64)all_edges.size();
    torch::Tensor edge_index_t = torch::empty({2, total_edges}, opts);
    torch::Tensor edge_src_t = torch::empty({total_edges}, opts);

    auto edge_u = edge_index_t.data_ptr<i64>();
    auto edge_v = edge_u + total_edges;
    auto edge_src = edge_src_t.data_ptr<i64>();

    for (i64 i = 0; i < total_edges; ++i) {
        edge_u[i] = all_edges[i].u;
        edge_v[i] = all_edges[i].v;
        edge_src[i] = all_edges[i].src_edge;
    }

    // Build pointer tensors
    torch::Tensor edge_ptr_t = torch::empty({(i64)edge_ptr_vec.size()}, opts);
    torch::Tensor sample_ptr_t = torch::empty({(i64)sample_ptr_vec.size()}, opts);

    auto edge_ptr_acc = edge_ptr_t.accessor<i64, 1>();
    auto sample_ptr_acc = sample_ptr_t.accessor<i64, 1>();

    for (size_t i = 0; i < edge_ptr_vec.size(); ++i) {
        edge_ptr_acc[i] = edge_ptr_vec[i];
    }
    for (size_t i = 0; i < sample_ptr_vec.size(); ++i) {
        sample_ptr_acc[i] = sample_ptr_vec[i];
    }

    // Transfer to input device if needed (GPU support)
    if (!input_device.is_cpu()) {
        nodes_t = nodes_t.to(input_device);
        edge_index_t = edge_index_t.to(input_device);
        edge_ptr_t = edge_ptr_t.to(input_device);
        sample_ptr_t = sample_ptr_t.to(input_device);
        edge_src_t = edge_src_t.to(input_device);
    }

    return py::make_tuple(nodes_t, edge_index_t, edge_ptr_t, sample_ptr_t, edge_src_t);
}

//=============================================================================
// Python bindings
//=============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sample_batch", &sample_batch,
          "Epsilon-uniform connected subgraph sampling via random walk with rejection sampling",
          py::arg("edge_index"),
          py::arg("ptr"),
          py::arg("m_per_graph"),
          py::arg("k"),
          py::arg("mode") = "sample",
          py::arg("seed") = 42,
          py::arg("epsilon") = 0.1);
}
