#pragma once
// cache.hpp
// LRU cache for preprocessing handles

#include <unordered_map>
#include <list>
#include <mutex>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <torch/extension.h>

using i64 = int64_t;

// Simple LRU cache for preprocessing handles
template<typename K, typename V>
class LRUCache {
public:
    LRUCache(size_t capacity) : capacity_(capacity) {}

    // Get value if exists, returns {found, value}
    std::pair<bool, V> get(const K& key) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = map_.find(key);
        if (it == map_.end()) {
            return {false, V()};
        }
        // Move to front (most recently used)
        items_.splice(items_.begin(), items_, it->second);
        return {true, it->second->second};
    }

    // Put key-value pair, returns evicted value if eviction happened
    std::pair<bool, V> put(const K& key, const V& value) {
        std::lock_guard<std::mutex> lock(mutex_);

        // Check if key already exists
        auto it = map_.find(key);
        if (it != map_.end()) {
            // Update value and move to front
            it->second->second = value;
            items_.splice(items_.begin(), items_, it->second);
            return {false, V()};
        }

        // Evict if at capacity
        std::pair<bool, V> evicted = {false, V()};
        if (items_.size() >= capacity_ && capacity_ > 0) {
            // Evict least recently used (back of list)
            auto last = items_.back();
            evicted = {true, last.second};
            map_.erase(last.first);
            items_.pop_back();
        }

        // Insert new item at front
        items_.push_front({key, value});
        map_[key] = items_.begin();

        return evicted;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return items_.size();
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        items_.clear();
        map_.clear();
    }

private:
    size_t capacity_;
    mutable std::mutex mutex_;
    std::list<std::pair<K, V>> items_;  // front = most recent
    std::unordered_map<K, typename std::list<std::pair<K, V>>::iterator> map_;
};

// Hash a graph's edge_index tensor for cache key
inline size_t hash_graph(const torch::Tensor& edge_index, i64 num_nodes) {
    TORCH_CHECK(edge_index.device().is_cpu(), "edge_index must be on CPU");
    TORCH_CHECK(edge_index.dtype() == torch::kInt64, "edge_index must be int64");

    const i64 m = edge_index.size(1);
    auto ei_ptr = edge_index.data_ptr<i64>();

    // Use FNV-1a hash
    size_t hash = 14695981039346656037ULL;

    // Hash num_nodes
    hash ^= (size_t)num_nodes;
    hash *= 1099511628211ULL;

    // Hash num_edges
    hash ^= (size_t)m;
    hash *= 1099511628211ULL;

    // Hash edge content (sample if too large)
    const i64 sample_stride = (m > 1000) ? (m / 500) : 1;
    for (i64 j = 0; j < m; j += sample_stride) {
        hash ^= (size_t)ei_ptr[j];
        hash *= 1099511628211ULL;
        hash ^= (size_t)ei_ptr[m + j];
        hash *= 1099511628211ULL;
    }

    return hash;
}
