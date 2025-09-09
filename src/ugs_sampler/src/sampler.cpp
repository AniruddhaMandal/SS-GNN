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
py::tuple sample(i64 handle, int m_per_graph, int k) {
    std::shared_ptr<Preproc> P;
    {
        std::lock_guard<std::mutex> lock(registry_mutex);
        auto it = registry.find(handle);
        if (it == registry.end()) throw std::runtime_error("Invalid preproc handle");
        P = it->second;
    }

    int B = m_per_graph;
    std::vector<std::vector<int>> samples(B);


    #pragma omp parallel for schedule(dynamic)
    for (int b=0; b<B; ++b) {
        ThreadRNG rng((uint64_t)(1234 + b));
        int root_vi, tries = 0;
        do {
            root_vi = rng.next_int((int)P->order.size());
            rand_grow_sample(*P, k, rng, samples[b], root_vi);
            ++tries;
        } while ((int)samples[b].size() < k && tries < 10);
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

        // edges
        for (auto &pr : samples_edges[b]) {
            edge_idx_ptr[epos]              = pr.first;
            edge_idx_ptr[total_edges+epos]  = pr.second;
            ++epos;
        }
        edge_ptr_ptr[b+1] = epos;
    }

    return py::make_tuple(nodes_t, edge_index_t, edge_ptr_t, graph_id_t);
}

// --- Pybind11 module ---
PYBIND11_MODULE(ugs_sampler, m) {
    m.doc() = "UGS sampler core (uses preprocessing from preproc.cpp)";
    m.def("sample", &sample, py::arg("handle"), py::arg("m_per_graph"), py::arg("k"),
          "Sample m_per_graph subgraphs of size k from preprocessed graph.");
    m.def("create_preproc", &create_preproc, py::arg("edge_index"), py::arg("num_nodes"), py::arg("k"),
          "Create preprocessing for a graph and return a handle (int).");
    m.def("destroy_preproc", &destroy_preproc, py::arg("handle"), "Destroy a preprocessing handle.");
    m.def("has_graphlets", &has_graphlets, py::arg("handle"), "Return true if preprocessed graph contains k-graphlets.");
    m.def("get_preproc_info", &get_preproc_info, py::arg("handle"), "Return small metadata dict for debugging.");
}
