#pragma once
// epsilon_uniform_sampler.hpp
// Epsilon-uniform connected subgraph sampler via random walk with rejection sampling

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <vector>
#include <cstdint>
#include <random>

namespace py = pybind11;
using i64 = int64_t;
using i32 = int32_t;

// Thread-safe RNG
struct EpsilonRNG {
    std::mt19937_64 gen;

    EpsilonRNG(uint64_t seed = 42) : gen(seed) {}

    inline uint64_t next_u64() {
        return gen();
    }

    inline int next_int(int n) {
        if (n <= 0) return 0;
        std::uniform_int_distribution<int> dist(0, n - 1);
        return dist(gen);
    }

    inline double next_double() {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        return dist(gen);
    }
};

// Function declarations
py::tuple sample_batch(
    const torch::Tensor& edge_index,
    const torch::Tensor& ptr,
    int m_per_graph,
    int k,
    const std::string& mode = "sample",
    uint64_t seed = 42,
    double epsilon = 0.1
);
