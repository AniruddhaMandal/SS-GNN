#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cstring>
#include "sampler.hpp"

namespace py = pybind11;
using i64 = int64_t;

// Return both: (renumbered per-graph edge_index, per-graph->global edge col map)
static std::pair<torch::Tensor, torch::Tensor>
slice_and_renumber_edge_index_with_map(const torch::Tensor& edge_index, i64 node_lo, i64 node_hi) {
    TORCH_CHECK(edge_index.device().is_cpu(), "edge_index must be on CPU");
    TORCH_CHECK(edge_index.dtype() == torch::kInt64, "edge_index must be int64");
    auto ei = edge_index.accessor<i64,2>();

    std::vector<i64> out_u; out_u.reserve(edge_index.size(1)/4);
    std::vector<i64> out_v; out_v.reserve(edge_index.size(1)/4);
    std::vector<i64> out_src; out_src.reserve(edge_index.size(1)/4);

    for (i64 j = 0; j < edge_index.size(1); ++j) {
        const i64 u = ei[0][j], v = ei[1][j];
        if (u >= node_lo && u < node_hi && v >= node_lo && v < node_hi) {
            out_u.push_back(u - node_lo);
            out_v.push_back(v - node_lo);
            out_src.push_back(j);                // global column index in batch.edge_index
        }
    }

    auto opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    const i64 E = (i64)out_u.size();

    torch::Tensor g_ei = torch::empty({2, E}, opts);
    auto gea = g_ei.accessor<i64,2>();
    for (i64 j = 0; j < E; ++j) {
        gea[0][j] = out_u[(size_t)j];
        gea[1][j] = out_v[(size_t)j];
    }

    torch::Tensor g_edge2global = torch::empty({E}, opts);
    auto gmap = g_edge2global.accessor<i64,1>();
    for (i64 j = 0; j < E; ++j) gmap[j] = out_src[(size_t)j];

    return {g_ei, g_edge2global};
}

py::tuple sample_batch(
    const torch::Tensor& edge_index,   // [2, E_total] global
    const torch::Tensor& ptr,          // [num_graphs+1]
    int m_per_graph,
    int k,
    const std::string& mode            // "sample" | "graph" | "global"
) {
    TORCH_CHECK(edge_index.device().is_cpu(), "edge_index must be on CPU");
    TORCH_CHECK(edge_index.dtype() == torch::kInt64, "edge_index must be int64");
    TORCH_CHECK(ptr.device().is_cpu(), "ptr must be on CPU");
    TORCH_CHECK(ptr.dtype() == torch::kInt64, "ptr must be int64");
    TORCH_CHECK(mode == "sample" || mode == "graph" || mode == "global",
                "mode must be one of: 'sample', 'graph', 'global'");

    const i64 num_graphs = ptr.size(0) - 1;
    const i64 B_total    = num_graphs * (i64)m_per_graph;

    auto opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU).pinned_memory(true);

    // outputs
    torch::Tensor nodes_out        = torch::full({B_total, k}, -1, opts);
    std::vector<i64> edge_ptr; edge_ptr.reserve((size_t)B_total + 1); edge_ptr.push_back(0);
    std::vector<i64> sample_ptr; sample_ptr.reserve((size_t)num_graphs + 1); sample_ptr.push_back(0);

    // chunks to concat at end
    std::vector<torch::Tensor> edge_chunks;
    std::vector<torch::Tensor> edge_src_global_chunks;
    edge_chunks.reserve((size_t)num_graphs);
    edge_src_global_chunks.reserve((size_t)num_graphs);

    auto ptr_acc = ptr.accessor<i64,1>();
    auto nodes_acc_out = nodes_out.accessor<i64,2>();

    i64 total_edges = 0;
    i64 Bpos = 0;

    for (i64 gi = 0; gi < num_graphs; ++gi) {
        const i64 lo = ptr_acc[gi], hi = ptr_acc[gi+1];
        const i64 n  = hi - lo;

        sample_ptr.push_back(sample_ptr.back() + m_per_graph);

        if (n <= 0 || n < k) {
            // fill degenerate samples
            for (int s = 0; s < m_per_graph; ++s) {
                const i64 out_idx = Bpos + s;
                for (i64 j = 0; j < k; ++j) nodes_acc_out[out_idx][j] = -1;
                edge_ptr.push_back(edge_ptr.back());
            }
            edge_chunks.push_back(torch::empty({2,0}, opts));
            edge_src_global_chunks.push_back(torch::empty({0}, opts));
            Bpos += m_per_graph;
            continue;
        }

        // slice per-graph edges + mapping to global columns
        auto [g_ei, g_edge2global] = slice_and_renumber_edge_index_with_map(edge_index, lo, hi);

        // build preproc and sample
        const i64 handle = create_preproc(g_ei, n, k);
        std::string edge_mode_for_sample;
        i64 base_offset = 0;
        if (mode == "sample") edge_mode_for_sample = "local";
        else if (mode == "graph") edge_mode_for_sample = "flat";
        else { edge_mode_for_sample = "global"; base_offset = lo; }

        // Expect: (nodes_t_local [m,k], edge_index_t_local [2,E_gi], edge_ptr_t_local [m+1], edge_src_idx_local [E_gi])
        py::tuple tup = sample(handle, m_per_graph, k, edge_mode_for_sample, base_offset);

        torch::Tensor nodes_t_local       = tup[0].cast<torch::Tensor>();
        torch::Tensor edge_index_t_local  = tup[1].cast<torch::Tensor>();
        torch::Tensor edge_ptr_t_local    = tup[2].cast<torch::Tensor>();
        torch::Tensor edge_src_idx_local  = tup[3].cast<torch::Tensor>();

        auto nodes_acc_local = nodes_t_local.accessor<i64,2>();
        auto epp = edge_ptr_t_local.accessor<i64,1>();

        // write nodes (already in correct coordinate system from sample())
        for (i64 b = 0; b < nodes_t_local.size(0); ++b) {
            const i64 out_idx = Bpos + b;
            for (i64 j = 0; j < k; ++j) {
                const i64 v = nodes_acc_local[b][j];
                // For 'global' mode, sample() already globalized nodes, so copy as-is
                // For 'sample' and 'graph' modes, we need to add lo offset
                nodes_acc_out[out_idx][j] = (v >= 0) ? (mode == "global" ? v : v + lo) : -1;
            }
        }

        // grow global edge_ptr by per-sample counts
        for (i64 b = 0; b < nodes_t_local.size(0); ++b) {
            const i64 cnt = epp[b+1] - epp[b];
            edge_ptr.push_back(edge_ptr.back() + cnt);
        }

        // Map edge_src_idx_local (per-graph edge columns) to global batch edge columns
        torch::Tensor edge_src_global_local;
        const i64 E_sampled = edge_index_t_local.size(1);
        
        if (edge_src_idx_local.defined() && edge_src_idx_local.numel() == E_sampled) {
            // edge_src_idx_local contains indices into g_ei (the per-graph edge_index)
            // g_edge2global maps g_ei columns -> batch.edge_index columns
            
            // Validate indices are in range
            auto src_acc = edge_src_idx_local.accessor<i64,1>();
            for (i64 i = 0; i < E_sampled; ++i) {
                if (src_acc[i] < 0 || src_acc[i] >= g_edge2global.size(0)) {
                    throw std::runtime_error(
                        "edge_src_idx_local[" + std::to_string(i) + "]=" + 
                        std::to_string(src_acc[i]) + " out of range [0," + 
                        std::to_string(g_edge2global.size(0)) + ")"
                    );
                }
            }
            
            edge_src_global_local = g_edge2global.index({edge_src_idx_local});
        } else {
            throw std::runtime_error(
                "edge_src_idx_local size mismatch: expected " + 
                std::to_string(E_sampled) + ", got " + 
                std::to_string(edge_src_idx_local.numel())
            );
        }

        total_edges += edge_index_t_local.size(1);
        edge_chunks.push_back(edge_index_t_local);
        edge_src_global_chunks.push_back(edge_src_global_local);

        destroy_preproc(handle);
        Bpos += nodes_t_local.size(0);
    }

    // concat edge chunks
    torch::Tensor edge_index_out = torch::empty({2, total_edges}, opts);
    auto out_u = edge_index_out.data_ptr<i64>();
    auto out_v = out_u + total_edges;

    i64 w = 0;
    for (const auto& chunk : edge_chunks) {
        const i64 E = chunk.size(1);
        if (E == 0) continue;
        auto in_u = chunk.data_ptr<i64>();
        auto in_v = in_u + E;
        std::memcpy(out_u + w, in_u, sizeof(i64)* (size_t)E);
        std::memcpy(out_v + w, in_v, sizeof(i64)* (size_t)E);
        w += E;
    }

    // concat edge_src_global chunks
    torch::Tensor edge_src_global_out = torch::empty({total_edges}, opts);
    auto esg = edge_src_global_out.data_ptr<i64>();
    w = 0;
    for (const auto& chunk : edge_src_global_chunks) {
        const i64 E = chunk.numel();
        if (E == 0) continue;
        std::memcpy(esg + w, chunk.data_ptr<i64>(), sizeof(i64)*(size_t)E);
        w += E;
    }

    // materialize ptr tensors
    torch::Tensor edge_ptr_out = torch::empty({(i64)edge_ptr.size()}, opts);
    auto epa = edge_ptr_out.accessor<i64,1>();
    for (size_t i = 0; i < edge_ptr.size(); ++i) epa[(i64)i] = edge_ptr[i];

    torch::Tensor sample_ptr_out = torch::empty({(i64)sample_ptr.size()}, opts);
    auto spa = sample_ptr_out.accessor<i64,1>();
    for (size_t i = 0; i < sample_ptr.size(); ++i) spa[(i64)i] = sample_ptr[i];

    // Return the minimal, attribute-ready pack
    return py::make_tuple(
        nodes_out,              // [B_total, k] (global ids)
        edge_index_out,         // [2, E_total]
        edge_ptr_out,           // [B_total+1]
        sample_ptr_out,         // [num_graphs+1]
        edge_src_global_out     // [E_total] maps into original batch.edge_index columns
    );
}