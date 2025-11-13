#pragma once
// uniform_sampler.hpp
// Truly uniform connected subgraph sampler via exhaustive enumeration

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <vector>
#include <cstdint>
#include <random>

namespace py = pybind11;
using i64 = int64_t;
using i32 = int32_t;

// Thread-safe RNG
struct UniformRNG {
    std::mt19937_64 gen;

    UniformRNG(uint64_t seed = 42) : gen(seed) {}

    inline uint64_t next_u64() {
        return gen();
    }

    inline int next_int(int n) {
        std::uniform_int_distribution<int> dist(0, n - 1);
        return dist(gen);
    }
};

// Single enumerated subgraph
struct Subgraph {
    std::vector<int> nodes;  // k node IDs

    Subgraph() = default;
    Subgraph(const std::vector<int>& n) : nodes(n) {}
};

// Enumerated subgraphs for one graph
struct EnumeratedSubgraphs {
    std::vector<Subgraph> subgraphs;
    int n;  // num nodes in graph
    int k;  // subgraph size

    EnumeratedSubgraphs() : n(0), k(0) {}
};

// Function declarations
py::tuple sample_batch(
    const torch::Tensor& edge_index,
    const torch::Tensor& ptr,
    int m_per_graph,
    int k,
    const std::string& mode = "sample",
    uint64_t seed = 42
);
