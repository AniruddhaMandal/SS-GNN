// ugs_sampler_batch_extension.cpp
// Added batch-aware sampling helper that respects graph boundaries in a batched PyG edge_index + ptr.
// Builds per-graph preprocessing handles using the existing create_preproc, calls sample(),
// then stitches the per-graph samples into a single batched return value.

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <memory>
#include <chrono>

#include "sampler.hpp"

// Helper: slice a global batched edge_index into a per-graph edge_index with renumbered nodes
static torch::Tensor slice_and_renumber_edge_index(const torch::Tensor &edge_index, i64 node_lo, i64 node_hi) {
    // edge_index: [2, M], CPU, int64
    TORCH_CHECK(edge_index.device().is_cpu(), "edge_index must be on CPU");
    TORCH_CHECK(edge_index.dtype() == torch::kInt64, "edge_index must be int64");

    auto ei = edge_index.accessor<i64,2>();
    std::vector<i64> out_u;
    std::vector<i64> out_v;
    out_u.reserve(edge_index.size(1)/4);
    out_v.reserve(edge_index.size(1)/4);

    for (i64 j = 0; j < edge_index.size(1); ++j) {
        i64 u = ei[0][j];
        i64 v = ei[1][j];
        if (u >= node_lo && u < node_hi && v >= node_lo && v < node_hi) {
            out_u.push_back(u - node_lo);
            out_v.push_back(v - node_lo);
        }
    }

    auto opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    torch::Tensor out = torch::empty({2, (i64)out_u.size()}, opts);
    auto out_acc = out.accessor<i64,2>();
    for (i64 j = 0; j < (i64)out_u.size(); ++j) {
        out_acc[0][j] = out_u[(size_t)j];
        out_acc[1][j] = out_v[(size_t)j];
    }
    return out;
}

// sample_batch(edge_index: LongTensor[2, M], ptr: LongTensor[num_graphs+1], m_per_graph: int, k: int)
// Returns (nodes_t, edge_index_t, edge_ptr_t, graph_id_t) in the same format as sample(), but aggregated for all graphs
py::tuple sample_batch(const torch::Tensor &edge_index, const torch::Tensor &ptr, int m_per_graph, int k) {
    TORCH_CHECK(edge_index.device().is_cpu(), "edge_index must be on CPU");
    TORCH_CHECK(edge_index.dtype() == torch::kInt64, "edge_index must be int64");
    TORCH_CHECK(ptr.device().is_cpu(), "ptr must be on CPU");
    TORCH_CHECK(ptr.dtype() == torch::kInt64, "ptr must be int64");

    int64_t num_graphs = (int64_t)ptr.size(0) - 1;
    std::vector<torch::Tensor> per_nodes; per_nodes.reserve(num_graphs);
    std::vector<torch::Tensor> per_edge_index; per_edge_index.reserve(num_graphs);
    std::vector<torch::Tensor> per_edge_ptr; per_edge_ptr.reserve(num_graphs);
    std::vector<torch::Tensor> per_graph_id; per_graph_id.reserve(num_graphs);

    int64_t total_B = num_graphs * (int64_t)m_per_graph;

    // We'll collect nodes and edges into vectors first (to compute total edges)
    std::vector<int64_t> nodes_flat; nodes_flat.resize(total_B * k, -1);
    std::vector<int64_t> graph_id_flat; graph_id_flat.resize(total_B, 0);

    std::vector<std::pair<int64_t,int64_t>> edges_all; // (u_local_index_in_total_nodes, v_local_index_in_total_nodes)
    std::vector<int64_t> edge_ptrs; edge_ptrs.reserve(total_B+1);
    edge_ptrs.push_back(0);

    int64_t Bpos = 0; // which sampled-subgraph index overall

    auto ptr_acc = ptr.accessor<i64,1>();

    for (int64_t gi = 0; gi < num_graphs; ++gi) {
        int64_t node_lo = ptr_acc[gi];
        int64_t node_hi = ptr_acc[gi+1];
        int64_t num_nodes = node_hi - node_lo;
        // --- add this right after computing node_lo, node_hi, num_nodes ---
        if (num_nodes < k) {
            // produce m_per_graph placeholder samples (nodes = all -1, no edges)
            for (int s = 0; s < m_per_graph; ++s) {
                int64_t out_idx = Bpos + s;
                // fill nodes_flat for this sample with -1
                for (int64_t j = 0; j < k; ++j) 
                    nodes_flat[out_idx * k + j] = -1;
                // set graph id
                graph_id_flat[out_idx] = gi;
                // no edges emitted: edge_ptr stays the same (push same value)
                edge_ptrs.push_back(edge_ptrs.back());
            }
            Bpos += m_per_graph;
            continue; // skip preprocessing / sampling for this graph
        }

        if (num_nodes <= 0) {
            // still advance graph_id positions and leave nodes as -1
            for (int s = 0; s < m_per_graph; ++s) {
                int64_t out_idx = Bpos + s;
                for (int64_t j = 0; j < k; ++j) nodes_flat[out_idx * k + j] = -1;
                graph_id_flat[Bpos] = gi;
                edge_ptrs.push_back(edge_ptrs.back());
            }
            Bpos += m_per_graph;
            continue;
        }
        // slice and renumber
        torch::Tensor g_ei = slice_and_renumber_edge_index(edge_index, node_lo, node_hi);
        // create preproc for this graph
        int64_t handle = create_preproc(g_ei, num_nodes, k);
        // sample m_per_graph subgraphs
        py::tuple tup = sample(handle, m_per_graph, k, "flat");
        // tup = (nodes_t, edge_index_t, edge_ptr_t, graph_id_t) where nodes_t contains local node ids (0..num_nodes-1)
        torch::Tensor nodes_t_local = tup[0].cast<torch::Tensor>();
        torch::Tensor edge_index_t_local = tup[1].cast<torch::Tensor>();
        torch::Tensor edge_ptr_t_local = tup[2].cast<torch::Tensor>();
        // convert nodes_t_local to global numbering and copy into nodes_flat
        auto nodes_acc = nodes_t_local.accessor<i64,2>();
        for (int64_t b = 0; b < nodes_t_local.size(0); ++b) {
            int64_t out_idx = Bpos + b;
            for (int64_t j = 0; j < k; ++j) {
                int64_t val = nodes_acc[b][j];
                if (val >= 0) nodes_flat[out_idx * k + j] = val + node_lo;
                else nodes_flat[out_idx * k + j] = -1;
            }
            graph_id_flat[out_idx] = gi;
        }
        // convert edges: edge_index_t_local has coords relative to each sample (0..k-1)
        auto edge_idx_acc = edge_index_t_local.accessor<i64,2>();
        auto edge_ptr_acc = edge_ptr_t_local.accessor<i64,1>();
        // For each sampled subgraph in this graph, determine its number of edges and compute global v indices
        for (int64_t local_b = 0; local_b < nodes_t_local.size(0); ++local_b) {
            int64_t start = edge_ptr_acc[local_b];
            int64_t end = edge_ptr_acc[local_b+1];
            for (int64_t epos = start; epos < end; ++epos) {
                int64_t u = edge_idx_acc[0][epos];
                int64_t v = edge_idx_acc[1][epos];
                // map u/v which are local indices 0..k-1 to global node positions within overall nodes_flat
                // Need to find which sample this edge belongs to: it's local_b, so global sample index is (Bpos + local_b)
                //int64_t sample_global_idx = Bpos + local_b;
                // For consistency with sampler.sample output, edges are stored as (node_idx_in_sample, node_idx_in_sample)
                // We'll store them the same way, but the nodes in nodes_flat carry global graph node ids.
                edges_all.emplace_back(u, v);
            }
            // update edge_ptrs
            edge_ptrs.push_back(edge_ptrs.back() + (end - start));
        }

        // cleanup
        destroy_preproc(handle);
        Bpos += nodes_t_local.size(0);
    }

    // Build final tensors
    auto opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU).pinned_memory(true);
    torch::Tensor nodes_out = torch::full({total_B, k}, -1, opts);
    // fill nodes
    auto nodes_out_acc = nodes_out.accessor<i64,2>();
    for (int64_t i = 0; i < total_B; ++i) {
        for (int64_t j = 0; j < k; ++j) nodes_out_acc[i][j] = nodes_flat[i*k + j];
    }

    int64_t total_edges = 0;
    for (auto &p : edges_all) total_edges++;
    torch::Tensor edge_index_out = torch::empty({2, total_edges}, opts);
    auto edge_out_acc = edge_index_out.accessor<i64,2>();
    for (int64_t e = 0; e < total_edges; ++e) {
        edge_out_acc[0][e] = edges_all[(size_t)e].first;
        edge_out_acc[1][e] = edges_all[(size_t)e].second;
    }

    torch::Tensor edge_ptr_out = torch::empty({(int64_t)edge_ptrs.size()}, opts);
    auto edge_ptr_acc_out = edge_ptr_out.accessor<i64,1>();
    for (size_t i = 0; i < edge_ptrs.size(); ++i) edge_ptr_acc_out[(i64)i] = edge_ptrs[i];

    torch::Tensor graph_id_out = torch::empty({total_B}, opts);
    auto graph_id_acc = graph_id_out.accessor<i64,1>();
    for (int64_t i = 0; i < total_B; ++i) graph_id_acc[i] = graph_id_flat[i];

    return py::make_tuple(nodes_out, edge_index_out, edge_ptr_out, graph_id_out);
}
