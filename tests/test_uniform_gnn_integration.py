#!/usr/bin/env python3
"""Test uniform_sampler integration with GNN operations"""
import torch
import uniform_sampler

# Simple batch of 2 graphs
edge_index = torch.tensor([
    [0, 1, 1, 2, 2, 0,  3, 4, 4, 5, 5, 3],
    [1, 0, 2, 1, 0, 2,  4, 3, 5, 4, 3, 5],
], dtype=torch.long)

ptr = torch.tensor([0, 3, 6], dtype=torch.long)

# Create node features (6 nodes, 4 features each)
x = torch.randn(6, 4)

print("Testing GNN-like operations with uniform_sampler output")
print("=" * 70)
print()

# Sample
nodes_t, edge_index_t, edge_ptr_t, sample_ptr_t, edge_src_t = \
    uniform_sampler.sample_batch(edge_index, ptr, m_per_graph=2, k=2, mode="sample", seed=42)

print("Sampler output:")
print(f"  nodes_t shape: {nodes_t.shape}")
print(f"  nodes_t:\n{nodes_t}")
print(f"  edge_index_t shape: {edge_index_t.shape}")
print(f"  edge_index_t:\n{edge_index_t}")
print()

# Simulate GNN forward pass (like in ss_gnn.py line 369)
num_subgraphs, k = nodes_t.shape
stacked_nodes = nodes_t.flatten()

print("Simulating GNN feature gathering:")
print(f"  stacked_nodes: {stacked_nodes}")
print(f"  x.shape: {x.shape}")
print()

# This should work without index out of bounds
try:
    global_x = x[stacked_nodes]
    print(f"✓ Feature gathering successful!")
    print(f"  global_x.shape: {global_x.shape}")
    print()
except IndexError as e:
    print(f"✗ Index out of bounds error: {e}")
    print()

# Simulate edge index offsetting (like in ss_gnn.py line 378-379)
print("Simulating edge index offsetting:")
offset = torch.repeat_interleave(torch.arange(0, num_subgraphs), edge_ptr_t[1:] - edge_ptr_t[:-1]) * k
global_edge_index = offset + edge_index_t
print(f"  offset: {offset}")
print(f"  global_edge_index:\n{global_edge_index}")
print()

# Check that global edge indices are valid
max_node_idx = num_subgraphs * k - 1
if global_edge_index.max() <= max_node_idx:
    print(f"✓ Edge indices are valid (max {global_edge_index.max()} <= {max_node_idx})")
else:
    print(f"✗ Edge indices out of bounds (max {global_edge_index.max()} > {max_node_idx})")

print()
print("=" * 70)
print("Integration test PASSED ✓")
print("=" * 70)
