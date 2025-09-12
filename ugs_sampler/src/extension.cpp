#include "sampler.hpp"

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
    m.def("sample_batch", &sample_batch, py::arg("edge_index"), py::arg("ptr"), py::arg("m_per_graph"), py::arg("k"),
          "Sample m_per_graph k-subgraphs per graph from a batched PyG edge_index + ptr.");
}