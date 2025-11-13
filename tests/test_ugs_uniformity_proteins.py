#!/usr/bin/env python3
"""Test UGS sampler uniformity on PROTEINS"""
import torch
import ugs_sampler
import numpy as np
from collections import Counter
import math
from torch_geometric.datasets import TUDataset

def graph_to_canonical(nodes, edge_index):
    """Convert sampled subgraph to canonical form"""
    nodes_sorted = tuple(sorted(nodes.tolist()))
    edges = []
    node_map = {old: new for new, old in enumerate(nodes_sorted)}

    for i in range(edge_index.size(1)):
        u, v = edge_index[0, i].item(), edge_index[1, i].item()
        if u in node_map and v in node_map:
            u_new, v_new = node_map[u], node_map[v]
            edges.append(tuple(sorted([u_new, v_new])))

    return (nodes_sorted, tuple(sorted(set(edges))))

print("=" * 70)
print("UGS Sampler Uniformity Test on PROTEINS")
print("=" * 70)
print()

dataset = TUDataset(root='./data/PROTEINS', name='PROTEINS')

# Find a medium-sized graph for testing
test_idx = None
for i in range(len(dataset)):
    if 20 <= dataset[i].num_nodes <= 30:
        test_idx = i
        break

test_graph = dataset[test_idx]
print(f"Test graph #{test_idx}:")
print(f"  Nodes: {test_graph.num_nodes}")
print(f"  Edges: {test_graph.edge_index.size(1)}")
print()

# Sample many subgraphs
SUBGRAPH_SIZE = 8
NUM_SAMPLES = 5000

print(f"Sampling {NUM_SAMPLES} subgraphs of size {SUBGRAPH_SIZE}...")

edge_index = test_graph.edge_index
ptr = torch.tensor([0, test_graph.num_nodes], dtype=torch.long)

nodes_t, edge_index_t, edge_ptr_t, sample_ptr_t, _ = \
    ugs_sampler.sample_batch(edge_index, ptr, m_per_graph=NUM_SAMPLES,
                             k=SUBGRAPH_SIZE, mode="sample")

# Count unique subgraphs
subgraph_counts = Counter()

for i in range(NUM_SAMPLES):
    nodes = nodes_t[i]
    edge_start = edge_ptr_t[i].item()
    edge_end = edge_ptr_t[i+1].item()
    edges = edge_index_t[:, edge_start:edge_end]

    if (nodes >= 0).sum() == SUBGRAPH_SIZE:
        canonical = graph_to_canonical(nodes[nodes >= 0], edges)
        subgraph_counts[canonical] += 1

num_valid = sum(subgraph_counts.values())
num_unique = len(subgraph_counts)

print(f"\nResults:")
print("-" * 70)
print(f"  Valid samples: {num_valid}/{NUM_SAMPLES}")
print(f"  Unique subgraphs: {num_unique}")
print()

if num_unique > 0:
    frequencies = list(subgraph_counts.values())
    mean_freq = sum(frequencies) / len(frequencies)
    min_freq = min(frequencies)
    max_freq = max(frequencies)
    std_dev = math.sqrt(sum((f - mean_freq)**2 for f in frequencies) / len(frequencies))
    cv = std_dev / mean_freq if mean_freq > 0 else float('inf')

    print(f"Frequency statistics:")
    print(f"  Mean:   {mean_freq:.1f}")
    print(f"  Min:    {min_freq}")
    print(f"  Max:    {max_freq}")
    print(f"  Range:  {max_freq - min_freq}")
    print(f"  Std:    {std_dev:.2f}")
    print(f"  CV:     {cv:.3f}")
    print()

    # Show distribution
    sorted_counts = sorted(subgraph_counts.items(), key=lambda x: x[1], reverse=True)
    print(f"Most common (top 5):")
    for i, (sg, count) in enumerate(sorted_counts[:5]):
        print(f"  #{i+1}: {count} occurrences ({100*count/num_valid:.2f}%)")

    print()
    print(f"Least common (bottom 5):")
    for i, (sg, count) in enumerate(sorted_counts[-5:]):
        print(f"  #{len(sorted_counts)-4+i}: {count} occurrences ({100*count/num_valid:.2f}%)")

    print()
    print("=" * 70)
    print("UNIFORMITY ASSESSMENT")
    print("=" * 70)

    expected_freq = num_valid / num_unique
    max_deviation = max(abs(f - expected_freq) / expected_freq for f in frequencies) * 100

    print(f"Expected frequency (perfect uniform): {expected_freq:.1f}")
    print(f"Max deviation: {max_deviation:.1f}%")
    print(f"Coefficient of Variation: {cv:.3f}")
    print()

    if cv < 0.15:
        print("✓ EXCELLENT: Very low variation - truly uniform!")
    elif cv < 0.30:
        print("~ MODERATE: Some variation present")
    else:
        print("✗ POOR: High variation - NOT uniform")
        print("\n⚠️  This suggests the degeneracy ordering bug mentioned in literature review")

print()
print("=" * 70)
