// sampler.cpp
// UGS sampling implementation — relies on preprocessing from preproc.cpp

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <vector>
#include <random>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <omp.h>

#include "sampler.hpp"   


// -------- Registry accessor (defined in preproc.cpp) --------
// extern declarations ensure we use the same registry defined in preproc.cpp


// --- Rand-Grow helper (same as before, trimmed) ---
static void rand_grow_sample(const Preproc &P, int k, ThreadRNG &rng,
                             std::vector<int> &out_seq, int root_vi) {
    int root = P.order[root_vi];
    out_seq.clear();
    out_seq.reserve(k);
    out_seq.push_back(root);

    for (int step=1; step<k; ++step) {
        std::vector<int> cut;
        for (int u : out_seq) {
            for (i64 p = P.indptr[u]; p < P.indptr[u+1]; ++p) {
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
py::tuple sample(i64 handle, int m_per_graph, int k, std::string edge_mode = "local") {
    std::shared_ptr<Preproc> P;
    {
        std::lock_guard<std::mutex> lock(registry_mutex);
        auto it = registry.find(handle);
        if (it == registry.end()) throw std::runtime_error("Invalid preproc handle");
        P = it->second;
    }

    int B = m_per_graph;
    std::vector<std::vector<int>> samples(B);

    std::vector<int> viable_roots;
    viable_roots.reserve(P->order.size());
    for (int vi = 0; vi < (int)P->order.size(); ++vi)
        if (P->bucket_b[vi] > 0.0) viable_roots.push_back(vi);

    // Relax if empty: suffix_deg > 0
    if (viable_roots.empty()) {
        for (int vi = 0; vi < (int)P->order.size(); ++vi)
            if (P->suffix_deg[vi] > 0) viable_roots.push_back(vi);
    }
    // Final relaxation: allow all positions
    if (viable_roots.empty()) {
        for (int vi = 0; vi < (int)P->order.size(); ++vi)
            viable_roots.push_back(vi);
    }

    if (viable_roots.empty()) {
        // nothing we can do
        std::fprintf(stderr, "sample: no viable_roots at all (n_pos=%zu, k=%d)\n", P->order.size(), k);
        throw std::runtime_error("No viable roots available");
    }


    for (int b=0; b<B; ++b) {
        if (viable_roots.empty()) throw std::runtime_error("No viable roots available"); // nothing to sample
        uint64_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count()^(uint64_t)b;
        ThreadRNG rng(seed);
        int root_vi, tries = 0;
        do {
            root_vi = viable_roots[rng.next_int((int)viable_roots.size())];
            rand_grow_sample(*P, k, rng, samples[b], root_vi);
            ++tries;
        } while ((int)samples[b].size() < k && tries < 100);
    }

    // gather edges
    std::vector<std::vector<std::pair<int,int>>> samples_edges(B);
    int64_t total_edges = 0;

    for (int b=0; b<B; ++b) {
        auto &nodes = samples[b];
        if ((int)nodes.size() < k) continue; // skip failed
        // global->local map
        std::unordered_map<int,int> g2l;
        for (int i=0;i<k;++i) g2l[nodes[i]] = i;

        for (int i=0;i<k;++i) {
            int u = nodes[i];
            for (i64 p = P->indptr[u]; p < P->indptr[u+1]; ++p) {
                int v = P->indices[p];
                auto it = g2l.find(v);
                if (it != g2l.end()) {
                    samples_edges[b].push_back({i, it->second});
                }
            }
        }
        total_edges += samples_edges[b].size();
    }

    auto opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU).pinned_memory(true);
    torch::Tensor nodes_t     = torch::full({B, k}, -1, opts);
    torch::Tensor edge_ptr_t  = torch::empty({B+1}, opts);
    torch::Tensor edge_index_t= torch::empty({2, total_edges}, opts);
    torch::Tensor graph_id_t  = torch::zeros({B}, opts);

    auto nodes_ptr = nodes_t.data_ptr<i64>();
    auto edge_ptr_ptr = edge_ptr_t.data_ptr<i64>();
    auto edge_idx_ptr = edge_index_t.data_ptr<i64>();

    edge_ptr_ptr[0] = 0;
    int64_t epos = 0;

    for (int b=0; b<B; ++b) {
        // nodes
        for (int i=0;i<(int)samples[b].size() && i<k;++i)
            nodes_ptr[b*k + i] = samples[b][i];
        for (int i=(int)samples[b].size(); i<k; ++i)
            nodes_ptr[b*k + i] = -1; // ensure remainder slots are -1

        // edges
        for (auto &pr : samples_edges[b]) {
            int u_local = pr.first;
            int v_local = pr.second;

            // Check the global node ids stored in nodes_ptr for validity.
            // If either endpoint is -1, skip emitting this edge.
            int64_t uglob = nodes_ptr[b*(int64_t)k + (int64_t)u_local];
            int64_t vglob = nodes_ptr[b*(int64_t)k + (int64_t)v_local];
            if (uglob == -1 || vglob == -1) {
                continue; // skip invalid edge referencing placeholder node
            }

            int64_t u_out = 0, v_out = 0;
            if (edge_mode == "local") {
                // keep local indices 0..k-1
                u_out = (int64_t)u_local;
                v_out = (int64_t)v_local;
            } else if (edge_mode == "flat") {
                // flattened indices in 0 .. (B*k - 1)
                u_out = (int64_t)b * (int64_t)k + (int64_t)u_local;
                v_out = (int64_t)b * (int64_t)k + (int64_t)v_local;
            } else if (edge_mode == "global") {
                // use global node ids that were written into nodes_ptr
                u_out = uglob;
                v_out = vglob;
            } else {
                throw std::runtime_error(std::string("Unknown edge_mode: ") + edge_mode);
            }

            // write edge (note we preallocated for the original total_edges; we may write fewer)
            edge_idx_ptr[epos]             = u_out;
            edge_idx_ptr[total_edges + epos] = v_out;
            ++epos;
        }
        edge_ptr_ptr[b+1] = epos;
    }

    // If we skipped any edges, trim the edge_index_t to the actual emitted size (epos).
    if (epos < total_edges) {
        // allocate a smaller tensor and copy the emitted prefix
        torch::Tensor edge_index_trim = torch::empty({2, epos}, opts);
        auto edge_trim_ptr = edge_index_trim.data_ptr<i64>();
        for (int64_t i = 0; i < epos; ++i) {
            edge_trim_ptr[i] = edge_idx_ptr[i];
            edge_trim_ptr[epos + i] = edge_idx_ptr[total_edges + i];
        }
        edge_index_t = edge_index_trim;
    }

    return py::make_tuple(nodes_t, edge_index_t, edge_ptr_t, graph_id_t);
}