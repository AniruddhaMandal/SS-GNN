#pragma once
// sampler.hpp
// Public C++ API declarations for the UGS sampler & preprocessing.
//
// Requirements:
//   - libtorch (torch/extension.h) for torch::Tensor
//   - pybind11 for py::tuple if you want to return python tuples directly
//
// NOTE: implementations live in sampler.cpp and preproc.cpp

#include <mutex>
#include <vector>
#include <memory>
#include <cstdint>
#include <unordered_map>
#include <torch/extension.h>
#include <pybind11/pybind11.h>


namespace py = pybind11;

using i64 = int64_t;
using i32 = int32_t;

// Thread-local RNG (simple xorshift) - defined before AliasTable
struct ThreadRNG {
    uint64_t s;
    ThreadRNG(uint64_t seed = 123456789ULL) { if (seed == 0) seed = 1; s = seed; }
    inline uint64_t next_u64() {
        uint64_t x = s;
        x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
        s = x;
        return x * 2685821657736338717ULL;
    }
    inline int next_int(int n) { return (int)(next_u64() % (uint64_t)n); }
};

// -- AliasTable (same small implementation used in sampler.cpp) --
struct AliasTable {
    std::vector<double> prob;
    std::vector<int> alias;
    int n = 0;
    AliasTable() = default;
    void build(const std::vector<double>& weights) {
        n = (int)weights.size();
        prob.assign(n, 0.0);
        alias.assign(n, 0);
        if (n == 0) return;
        double sum = 0.0;
        for (double w : weights) sum += w;
        std::vector<double> p(n);
        for (int i=0;i<n;++i) p[i] = weights[i] * n / (sum > 0 ? sum : 1.0);
        std::vector<int> small, large;
        small.reserve(n); large.reserve(n);
        for (int i=0;i<n;++i) {
            if (p[i] < 1.0) small.push_back(i);
            else large.push_back(i);
        }
        while (!small.empty() && !large.empty()) {
            int s = small.back(); small.pop_back();
            int l = large.back();
            prob[s] = p[s];
            alias[s] = l;
            p[l] = (p[l] + p[s]) - 1.0;
            if (p[l] < 1.0) { small.push_back(l); large.pop_back(); }
        }
        for (int idx : large) prob[idx] = 1.0;
        for (int idx : small) prob[idx] = 1.0;
    }

    // Sample from the alias table using weighted distribution
    int sample(ThreadRNG& rng) const {
        if (n == 0) return -1;
        int i = rng.next_int(n);
        double u = (double)rng.next_u64() / (double)UINT64_MAX;
        return (u < prob[i]) ? i : alias[i];
    }
};

// -- Preproc struct (compatible with sampler.cpp) --
struct Preproc {
    i64 n = 0;
    i64 m = 0;
    std::vector<i64> indptr;   // length n+1
    std::vector<i32> indices;  // neighbors (undirected stored as both directions)
    std::vector<i32> edge_col_of_csr_pos;  // size == indices.size()
    std::vector<int> order;    // ≺ order (vertex ids)
    std::vector<int> index_of; // inverse mapping: index_of[vertex] -> position in order
    std::vector<i32> suffix_deg; // degree inside suffix (indexed by position in order)
    std::vector<double> bucket_b; // weights per order-position (b[vi])
    AliasTable alias;
    double Z = 0.0;
};

// Declaration for batch_extension
py::tuple sample_batch(const torch::Tensor &edge_index, const torch::Tensor &ptr, int m_per_graph, int k,  const std::string &mode); 

// Declaretion for sampler 
py::tuple sample(i64 handle, int m_per_graph, int k, std::string edge_mode, int64_t base_offset);

// Declaretion for preproc functions  
int64_t create_preproc(const torch::Tensor &edge_index, i64 num_nodes, int k);
void destroy_preproc(int64_t handle);
bool has_graphlets(int64_t handle);
py::dict get_preproc_info(int64_t handle);

// Registry: handle -> Preproc shared_ptr
extern std::mutex registry_mutex;
extern std::unordered_map<i64, std::shared_ptr<Preproc>> registry;