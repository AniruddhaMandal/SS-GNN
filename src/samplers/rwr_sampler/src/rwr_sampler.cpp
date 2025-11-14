// rwr_sampler.cpp
// Random-Walk-with-Restart connected induced k-subgraph sampler
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <random>
#include <omp.h>

namespace py = pybind11;
using i64 = int64_t;

// Small fast RNG wrapper (thread-local)
struct SplitMix64 {
    uint64_t state;
    SplitMix64(uint64_t seed=0x9e3779b97f4a7c15ULL){ state = seed + 0x9e3779b97f4a7c15ULL; }
    uint64_t next_u64() {
        uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
    double next_double() { return (next_u64() >> 11) * (1.0/9007199254740992.0); }
    int next_int(int bound) { return (int)(next_u64() % (uint64_t)bound); }
};

// Build adjacency per graph (0-indexed local)
static void build_adj_per_graph(
    const torch::Tensor& edge_index_cpu,
    const torch::Tensor& ptr_cpu,
    std::vector<std::vector<std::vector<int>>>& adjs
) {
    auto ei = edge_index_cpu.accessor<i64,2>();
    auto ptr = ptr_cpu.accessor<i64,1>();
    i64 num_graphs = ptr.size(0) - 1;
    i64 m = edge_index_cpu.size(1);

    adjs.resize((size_t)num_graphs);

    // Pre-size adjacency lists
    for (i64 g = 0; g < num_graphs; ++g) {
        int n = (int)(ptr[g+1] - ptr[g]);
        adjs[(size_t)g].assign(n, {});
    }

    // Fill edges
    for (i64 e = 0; e < m; ++e) {
        i64 u = ei[0][e];
        i64 v = ei[1][e];

        // find graph by ptr: binary search could be used, but ptr is small; we'll do linear scan per edge (still fine)
        // Optimize by scanning graphs and filling edges if they fall into graph range
        // We'll do a simple approach: iterate graphs and check range (fast enough for batched graphs)
        // For speed, we can later optimize by mapping node->graph, but keep code simple and correct.
        for (i64 g = 0; g < num_graphs; ++g) {
            i64 start = ptr[g];
            i64 end = ptr[g+1];
            if (u >= start && u < end && v >= start && v < end) {
                int u_local = (int)(u - start);
                int v_local = (int)(v - start);
                adjs[(size_t)g][u_local].push_back(v_local);
                adjs[(size_t)g][v_local].push_back(u_local);
                break;
            }
        }
    }
}

// RWR sampling per graph
py::tuple sample_batch(
    const torch::Tensor& edge_index,
    const torch::Tensor& ptr,
    int m_per_graph,
    int k,
    const std::string& mode = "sample",
    uint64_t seed = 42,
    double p_restart = 0.2
) {
    TORCH_CHECK(edge_index.dtype() == torch::kInt64, "edge_index must be int64");
    TORCH_CHECK(ptr.dtype() == torch::kInt64, "ptr must be int64");
    TORCH_CHECK(k >= 1, "k must be >= 1");
    TORCH_CHECK(p_restart >= 0.0 && p_restart <= 1.0, "p_restart in [0,1]");

    auto input_device = edge_index.device();

    // CPU copies
    auto edge_index_cpu = edge_index.cpu();
    auto ptr_cpu = ptr.cpu();

    auto ptr_acc = ptr_cpu.accessor<i64,1>();
    const i64 num_graphs = ptr_acc.size(0) - 1;

    // Build per-graph adjacency lists
    std::vector<std::vector<std::vector<int>>> adjs;
    build_adj_per_graph(edge_index_cpu, ptr_cpu, adjs);

    // Prepare outputs
    const i64 total_samples = num_graphs * (i64)m_per_graph;
    auto opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU).pinned_memory(true);
    torch::Tensor nodes_t = torch::full({total_samples, k}, -1, opts);

    std::vector<i64> edge_ptr_vec;
    edge_ptr_vec.reserve((size_t)total_samples + 1);
    edge_ptr_vec.push_back(0);

    std::vector<i64> sample_ptr_vec;
    sample_ptr_vec.reserve((size_t)num_graphs + 1);
    sample_ptr_vec.push_back(0);

    struct EdgeData { i64 u; i64 v; i64 src_edge; };
    std::vector<EdgeData> all_edges;
    all_edges.reserve(1024);

    auto nodes_acc = nodes_t.accessor<i64,2>();

    i64 sample_idx = 0;

    // Parallel over graphs
    #pragma omp parallel for schedule(dynamic)
    for (i64 g = 0; g < num_graphs; ++g) {
        // Per-thread local containers to avoid locking
        std::vector<i64> local_nodes_flat; // will write to global tensor with atomic-like approach later
        std::vector<EdgeData> local_edges;
        std::vector<i64> local_edge_ptrs;
        local_edge_ptrs.reserve(m_per_graph);

        SplitMix64 rng(seed + (uint64_t)g + (uint64_t)omp_get_thread_num()*104729);

        int n = (int)adjs[(size_t)g].size();
        i64 node_offset = ptr_acc[g];

        // If graph has no nodes, emit empty samples
        if (n == 0) {
            #pragma omp critical
            {
                for (int s = 0; s < m_per_graph; ++s) {
                    edge_ptr_vec.push_back(edge_ptr_vec.back());
                    sample_idx++;
                }
                sample_ptr_vec.push_back(sample_idx);
            }
            continue;
        }

        // If graph has fewer nodes than k, emit dummy samples (nodes_t already initialized with -1)
        if (n < k) {
            #pragma omp critical
            {
                for (int s = 0; s < m_per_graph; ++s) {
                    edge_ptr_vec.push_back(edge_ptr_vec.back());
                    sample_idx++;
                }
                sample_ptr_vec.push_back(sample_idx);
            }
            continue;
        }

        // For each sample
        for (int s = 0; s < m_per_graph; ++s) {
            // RWR sampling
            int seed_node = rng.next_int(n);
            int cur = seed_node;
            std::vector<int> chosen_nodes;
            chosen_nodes.reserve(k);
            std::vector<char> in_set(n, 0);

            // add seed
            chosen_nodes.push_back(cur);
            in_set[cur] = 1;

            // Iteration limit to prevent infinite loops on disconnected components
            int max_iterations = n * k * 10;  // Generous limit for pathological cases
            int iterations = 0;

            while ((int)chosen_nodes.size() < k && iterations < max_iterations) {
                iterations++;
                double r = rng.next_double();
                if (r < p_restart || adjs[(size_t)g][cur].empty()) {
                    cur = seed_node;
                } else {
                    const auto &nbrs = adjs[(size_t)g][cur];
                    cur = nbrs[rng.next_int((int)nbrs.size())];
                }
                if (!in_set[cur]) {
                    in_set[cur] = 1;
                    chosen_nodes.push_back(cur);
                }
            }

            // Check if sampling failed (couldn't get k distinct nodes)
            if ((int)chosen_nodes.size() < k) {
                // Emit dummy sample (nodes_t already initialized with -1)
                #pragma omp critical
                {
                    edge_ptr_vec.push_back(edge_ptr_vec.back());
                    sample_idx++;
                }
                continue;
            }

            // write chosen nodes into global tensor (global IDs)
            #pragma omp critical
            {
                for (int i = 0; i < k; ++i) {
                    i64 node_global = node_offset + (i64)chosen_nodes[i];
                    nodes_acc[sample_idx][i] = node_global;
                }
            }

            // Extract edges among chosen nodes: scan over adjacency for chosen nodes
            // Build a set for O(1) membership
            std::unordered_set<int> node_set;
            node_set.reserve(chosen_nodes.size()*2);
            for (int x : chosen_nodes) node_set.insert(x);

            // Collect local edges (we will output either local indices or global depending on mode)
            for (int u_local : chosen_nodes) {
                for (int v_local : adjs[(size_t)g][u_local]) {
                    if (node_set.count(v_local)) {
                        i64 u_out, v_out;
                        if (mode == "sample") {
                            // map graph local -> sample local index
                            int sample_idx_local = -1;
                            for (int i = 0; i < (int)chosen_nodes.size(); ++i) {
                                if (chosen_nodes[i] == u_local) { sample_idx_local = i; break; }
                            }
                            int sample_idx_local_v = -1;
                            for (int i = 0; i < (int)chosen_nodes.size(); ++i) {
                                if (chosen_nodes[i] == v_local) { sample_idx_local_v = i; break; }
                            }
                            if (sample_idx_local == -1 || sample_idx_local_v == -1) continue;
                            u_out = sample_idx_local;
                            v_out = sample_idx_local_v;
                        } else {
                            u_out = node_offset + u_local;
                            v_out = node_offset + v_local;
                        }
                        // to avoid duplicates (u,v) and (v,u) both added, we'll add all and later dedupe when writing tensors
                        local_edges.push_back({u_out, v_out, /*src_edge*/ -1});
                    }
                }
            }

            // record local edge count pointer
            #pragma omp critical
            {
                all_edges.insert(all_edges.end(), local_edges.begin(), local_edges.end());
                edge_ptr_vec.push_back((i64)all_edges.size());
                local_edges.clear();
                sample_idx++;
            }
        } // end samples per graph

        #pragma omp critical
        {
            sample_ptr_vec.push_back(sample_idx);
        }
    } // end parallel graphs

    // Build edge_index_t (2 x E) and edge_src_t (E)
    i64 total_edges = (i64)all_edges.size();
    torch::Tensor edge_index_t = torch::empty({2, total_edges}, opts);
    torch::Tensor edge_src_t = torch::empty({total_edges}, opts);

    auto edge_u = edge_index_t.data_ptr<i64>();
    auto edge_v = edge_u + total_edges;
    auto edge_src = edge_src_t.data_ptr<i64>();

    for (i64 i = 0; i < total_edges; ++i) {
        edge_u[i] = all_edges[(size_t)i].u;
        edge_v[i] = all_edges[(size_t)i].v;
        edge_src[i] = all_edges[(size_t)i].src_edge;
    }

    // Build pointer tensors
    torch::Tensor edge_ptr_t = torch::empty({(i64)edge_ptr_vec.size()}, opts);
    torch::Tensor sample_ptr_t = torch::empty({(i64)sample_ptr_vec.size()}, opts);
    auto edge_ptr_acc = edge_ptr_t.accessor<i64,1>();
    auto sample_ptr_acc = sample_ptr_t.accessor<i64,1>();
    for (size_t i = 0; i < edge_ptr_vec.size(); ++i) edge_ptr_acc[(i64)i] = edge_ptr_vec[i];
    for (size_t i = 0; i < sample_ptr_vec.size(); ++i) sample_ptr_acc[(i64)i] = sample_ptr_vec[i];

    // Move to input device if necessary
    if (!input_device.is_cpu()) {
        nodes_t = nodes_t.to(input_device);
        edge_index_t = edge_index_t.to(input_device);
        edge_ptr_t = edge_ptr_t.to(input_device);
        sample_ptr_t = sample_ptr_t.to(input_device);
        edge_src_t = edge_src_t.to(input_device);
    }

    return py::make_tuple(nodes_t, edge_index_t, edge_ptr_t, sample_ptr_t, edge_src_t);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sample_batch", &sample_batch,
          "Random-Walk-with-Restart (RWR) connected induced subgraph sampler",
          py::arg("edge_index"),
          py::arg("ptr"),
          py::arg("m_per_graph"),
          py::arg("k"),
          py::arg("mode") = "sample",
          py::arg("seed") = 42,
          py::arg("p_restart") = 0.2);
}
