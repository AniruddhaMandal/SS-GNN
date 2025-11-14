#pragma once

#include <torch/extension.h>
#include <vector>
#include <random>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <cmath>

// RNG structure for consistent random number generation
struct ApxRNG {
    std::mt19937_64 gen;

    ApxRNG(uint64_t seed = 42) : gen(seed) {}

    inline int next_int(int n) {
        return std::uniform_int_distribution<int>(0, n - 1)(gen);
    }

    inline double next_double() {
        return std::uniform_real_distribution<double>(0.0, 1.0)(gen);
    }
};

// Graph structure for efficient access
struct Graph {
    int n;  // number of vertices
    std::vector<std::vector<int>> adj;  // adjacency lists
    std::vector<int> degrees;

    Graph(const torch::Tensor& edge_index, const torch::Tensor& ptr);

    inline int degree(int v) const { return degrees[v]; }
    inline const std::vector<int>& neighbors(int v) const { return adj[v]; }
    inline bool has_edge(int u, int v) const;
};

// Degree-dominating order structure
struct DDOrder {
    std::vector<int> order;  // vertex ordering
    std::vector<int> position;  // position[v] = position of v in order
    std::vector<double> estimates;  // bucket size estimates

    DDOrder(int n) : order(n), position(n), estimates(n, 0.0) {}

    inline bool comes_before(int u, int v) const {
        return position[u] < position[v];
    }
};

// Forward declarations of main algorithm functions

// Algorithm 5: APX-DD - Approximate degree-dominating order
DDOrder apx_dd(const Graph& G, int k, double beta, ApxRNG& rng);

// Algorithm 7: EstimateCuts - Estimate cut sizes
std::vector<double> estimate_cuts(
    const Graph& G,
    const DDOrder& order,
    int v,
    const std::vector<int>& U,
    int k,
    double alpha,
    double beta,
    double delta,
    ApxRNG& rng
);

// Algorithm 8: APX-RAND-GROW - Random growing with estimated cuts
std::vector<int> apx_rand_grow(
    const Graph& G,
    const DDOrder& order,
    int v,
    int k,
    double alpha,
    double beta,
    double gamma,
    ApxRNG& rng
);

// Algorithm 9: APX-PROB - Approximate probability computation
double apx_prob(
    const Graph& G,
    const DDOrder& order,
    const std::vector<int>& S,
    double alpha,
    double beta,
    double rho,
    ApxRNG& rng
);

// Algorithm 10: APX-UGS - Main sampling algorithm
std::vector<int> apx_ugs_sample_one(
    const Graph& G,
    const DDOrder& order,
    int k,
    double epsilon,
    ApxRNG& rng
);

// Main entry point matching uniform_sampler API
py::tuple sample_batch(
    const torch::Tensor& edge_index,
    const torch::Tensor& ptr,
    int m_per_graph,
    int k,
    const std::string& mode = "sample",
    uint64_t seed = 42,
    double epsilon = 0.1
);
