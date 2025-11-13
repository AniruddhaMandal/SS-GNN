// uniform_sampler.cpp
// Truly uniform connected subgraph sampler via exhaustive enumeration

#include "uniform_sampler.hpp"
#include <algorithm>
#include <unordered_set>
#include <queue>

namespace py = pybind11;

//=============================================================================
// Helper: Check if a subset of nodes forms a connected induced subgraph
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
// Enumerate all connected k-subgraphs via DFS over combinations
//=============================================================================
static void enumerate_subgraphs(
    int n,
    int k,
    const std::vector<std::vector<int>>& adj,
    std::vector<Subgraph>& output
) {
    output.clear();

    // DFS to generate combinations
    std::vector<int> current;
    current.reserve(k);

    std::function<void(int)> dfs = [&](int start) {
        if ((int)current.size() == k) {
            // Check connectivity
            if (is_connected(current, adj)) {
                output.emplace_back(current);
            }
            return;
        }

        // Prune: not enough nodes left
        int needed = k - (int)current.size();
        if (n - start < needed) return;

        for (int v = start; v < n; ++v) {
            current.push_back(v);
            dfs(v + 1);
            current.pop_back();
        }
    };

    dfs(0);
}

//=============================================================================
// Main sampling function: sample_batch
//=============================================================================
py::tuple sample_batch(
    const torch::Tensor& edge_index,
    const torch::Tensor& ptr,
    int m_per_graph,
    int k,
    const std::string& mode,
    uint64_t seed
) {
    TORCH_CHECK(edge_index.device().is_cpu(), "edge_index must be on CPU");
    TORCH_CHECK(edge_index.dtype() == torch::kInt64, "edge_index must be int64");
    TORCH_CHECK(ptr.device().is_cpu(), "ptr must be on CPU");
    TORCH_CHECK(ptr.dtype() == torch::kInt64, "ptr must be int64");

    auto ei = edge_index.accessor<i64, 2>();
    auto ptr_acc = ptr.accessor<i64, 1>();

    const i64 num_graphs = ptr_acc.size(0) - 1;
    const i64 m = edge_index.size(1);

    // Build per-graph adjacency lists
    std::vector<EnumeratedSubgraphs> graph_subgraphs(num_graphs);

    for (i64 g = 0; g < num_graphs; ++g) {
        i64 node_start = ptr_acc[g];
        i64 node_end = ptr_acc[g + 1];
        int n = (int)(node_end - node_start);

        graph_subgraphs[g].n = n;
        graph_subgraphs[g].k = k;

        // Build local adjacency list (0-indexed within graph)
        std::vector<std::vector<int>> adj(n);

        for (i64 e = 0; e < m; ++e) {
            i64 u_global = ei[0][e];
            i64 v_global = ei[1][e];

            // Check if edge belongs to this graph
            if (u_global >= node_start && u_global < node_end &&
                v_global >= node_start && v_global < node_end) {
                int u_local = (int)(u_global - node_start);
                int v_local = (int)(v_global - node_start);

                adj[u_local].push_back(v_local);
                adj[v_local].push_back(u_local);
            }
        }

        // Enumerate all connected k-subgraphs
        enumerate_subgraphs(n, k, adj, graph_subgraphs[g].subgraphs);
    }

    // Sample m_per_graph subgraphs from each graph
    UniformRNG rng(seed);

    const i64 total_samples = num_graphs * m_per_graph;

    // Output tensors
    auto opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);

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

    for (i64 g = 0; g < num_graphs; ++g) {
        const auto& enum_sg = graph_subgraphs[g];
        i64 node_offset = ptr_acc[g];

        // Check if we have any subgraphs
        if (enum_sg.subgraphs.empty()) {
            // No valid subgraphs - fill with -1 (already initialized)
            for (int s = 0; s < m_per_graph; ++s) {
                edge_ptr_vec.push_back(edge_ptr_vec.back());
                sample_idx++;
            }
            sample_ptr_vec.push_back(sample_idx);
            continue;
        }

        // Sample m_per_graph times
        for (int s = 0; s < m_per_graph; ++s) {
            // Uniformly sample one subgraph
            int sg_idx = rng.next_int((int)enum_sg.subgraphs.size());
            const auto& sg = enum_sg.subgraphs[sg_idx];

            // Build local index map (graph-local to sample-local)
            std::unordered_map<int, int> graph_local_to_sample_local;
            for (int i = 0; i < k; ++i) {
                graph_local_to_sample_local[sg.nodes[i]] = i;
            }
            std::unordered_set<int> node_set(sg.nodes.begin(), sg.nodes.end());

            // Write nodes - ALWAYS use global node IDs (for feature gathering)
            for (int i = 0; i < k; ++i) {
                i64 node_global = node_offset + sg.nodes[i];
                nodes_acc[sample_idx][i] = node_global;
            }

            // Extract edges within subgraph
            for (i64 e = 0; e < m; ++e) {
                i64 u_global = ei[0][e];
                i64 v_global = ei[1][e];

                // Check if edge belongs to this graph
                if (u_global >= node_offset && u_global < node_offset + enum_sg.n &&
                    v_global >= node_offset && v_global < node_offset + enum_sg.n) {
                    int u_graph_local = (int)(u_global - node_offset);
                    int v_graph_local = (int)(v_global - node_offset);

                    // Check if both endpoints in subgraph
                    if (node_set.count(u_graph_local) && node_set.count(v_graph_local)) {
                        i64 u_out, v_out;

                        if (mode == "sample") {
                            // Local indexing within sample
                            u_out = graph_local_to_sample_local[u_graph_local];
                            v_out = graph_local_to_sample_local[v_graph_local];
                        } else {
                            // Global indexing
                            u_out = node_offset + u_graph_local;
                            v_out = node_offset + v_graph_local;
                        }

                        all_edges.push_back({u_out, v_out, e});
                    }
                }
            }

            edge_ptr_vec.push_back((i64)all_edges.size());
            sample_idx++;
        }

        sample_ptr_vec.push_back(sample_idx);
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

    return py::make_tuple(nodes_t, edge_index_t, edge_ptr_t, sample_ptr_t, edge_src_t);
}

//=============================================================================
// Python bindings
//=============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sample_batch", &sample_batch,
          "Truly uniform connected subgraph sampling via exhaustive enumeration",
          py::arg("edge_index"),
          py::arg("ptr"),
          py::arg("m_per_graph"),
          py::arg("k"),
          py::arg("mode") = "sample",
          py::arg("seed") = 42);
}
