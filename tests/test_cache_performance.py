#!/usr/bin/env python3
"""Test preprocessing cache performance

Run with: UGS_DEBUG=1 python tests/test_cache_performance.py
"""
import torch
import ugs_sampler
import time

print("=" * 60)
print("Cache Performance Test")
print("=" * 60)

# Create a batch with 10 identical graphs (should get cache hits)
print("\nTest 1: Batch with 10 identical graphs (expect cache hits)")
print("-" * 60)

# Single graph: triangle with 4 nodes
edge_index = torch.tensor([
    [0, 1, 1, 2, 2, 0],
    [1, 0, 2, 1, 0, 2],
], dtype=torch.long)

# Repeat the same graph 10 times with offset
graphs = []
for i in range(10):
    offset = i * 4
    graphs.append(edge_index + offset)

# Build batch
batch_edge_index = torch.cat(graphs, dim=1)
batch_ptr = torch.arange(0, 41, 4, dtype=torch.long)  # [0, 4, 8, 12, ..., 40]

# First call: all misses
start = time.time()
result1 = ugs_sampler.sample_batch(batch_edge_index, batch_ptr, m_per_graph=3, k=3, mode="sample")
time1 = time.time() - start
print(f"First call: {time1*1000:.2f}ms (building cache)")

# Second call: should hit cache
start = time.time()
result2 = ugs_sampler.sample_batch(batch_edge_index, batch_ptr, m_per_graph=3, k=3, mode="sample")
time2 = time.time() - start
print(f"Second call: {time2*1000:.2f}ms (using cache)")

speedup = time1 / time2 if time2 > 0 else 0
print(f"Speedup: {speedup:.2f}x")

# Third call: should still hit cache
start = time.time()
result3 = ugs_sampler.sample_batch(batch_edge_index, batch_ptr, m_per_graph=3, k=3, mode="sample")
time3 = time.time() - start
print(f"Third call: {time3*1000:.2f}ms (using cache)")

print("\n" + "=" * 60)
print("Cache is working if second/third calls are much faster!")
print("Run with UGS_DEBUG=1 to see cache hit rates")
print("=" * 60)
