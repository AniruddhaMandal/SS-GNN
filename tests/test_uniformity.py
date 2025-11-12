#!/usr/bin/env python3
"""Test if UGS sampler produces uniform samples on real QM9 graphs

Run with: UGS_DEBUG=1 python tests/test_uniformity.py
"""
import torch
import ugs_sampler
from collections import Counter
import sys

# Try to import QM9 dataset
try:
    from torch_geometric.datasets import QM9
    import os
    HAS_QM9 = True
except ImportError:
    HAS_QM9 = False
    print("PyTorch Geometric not available, using synthetic graph")

def graph_to_canonical(nodes, edge_index):
    """Convert sampled subgraph to canonical form for counting"""
    nodes_sorted = tuple(sorted(nodes.tolist()))

    # Build edge list with canonical node ids
    edges = []
    node_map = {old: new for new, old in enumerate(nodes_sorted)}

    for i in range(edge_index.size(1)):
        u, v = edge_index[0, i].item(), edge_index[1, i].item()
        if u in node_map and v in node_map:
            u_new, v_new = node_map[u], node_map[v]
            edges.append(tuple(sorted([u_new, v_new])))

    return (nodes_sorted, tuple(sorted(set(edges))))

print("=" * 60)
print("Testing Uniformity of UGS Sampler on Real Graph")
print("=" * 60)

# Load a real graph from QM9 or use synthetic
if HAS_QM9:
    print("\nLoading QM9 dataset...")
    dataset = QM9(root='/tmp/QM9')

    # Pick a medium-sized molecule (not too small, not too large)
    graph_idx = 100
    data = dataset[graph_idx]

    edge_index = data.edge_index
    num_nodes = data.num_nodes

    print(f"Using QM9 molecule {graph_idx}: {num_nodes} nodes, {edge_index.size(1)} edges")
else:
    # Synthetic graph: benzene-like ring
    print("\nUsing synthetic benzene-like graph (6 nodes in ring)")
    num_nodes = 6
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0],  # ring
        [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 0, 5],
    ], dtype=torch.long)

# Sample parameters
k = 4  # subgraph size
num_samples = 5000  # more samples = better uniformity test

print(f"\nSampling {num_samples} subgraphs of size k={k}")
print("This may take a moment...")

# Prepare batch (single graph, many samples)
ptr = torch.tensor([0, num_nodes], dtype=torch.long)

# Sample many times
nodes_t, edge_index_t, edge_ptr_t, sample_ptr_t, edge_src_global_t = \
    ugs_sampler.sample_batch(edge_index, ptr, m_per_graph=num_samples, k=k, mode="sample")

# Count unique subgraphs
print("\nAnalyzing samples...")
subgraph_counts = Counter()

for i in range(num_samples):
    nodes = nodes_t[i]
    # Get edges for this sample
    edge_start = edge_ptr_t[i].item()
    edge_end = edge_ptr_t[i+1].item()
    edges = edge_index_t[:, edge_start:edge_end]

    # Skip incomplete samples (nodes with -1)
    if (nodes >= 0).sum() < k:
        continue

    # Convert to canonical form
    canonical = graph_to_canonical(nodes[nodes >= 0], edges)
    subgraph_counts[canonical] += 1

# Analysis
num_valid_samples = sum(subgraph_counts.values())
num_unique = len(subgraph_counts)
incomplete_samples = num_samples - num_valid_samples

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"Total samples requested: {num_samples}")
print(f"Valid samples: {num_valid_samples}")
print(f"Incomplete samples: {incomplete_samples}")
print(f"Unique subgraphs found: {num_unique}")

if num_unique > 0:
    # Show frequency distribution
    frequencies = list(subgraph_counts.values())
    mean_freq = sum(frequencies) / len(frequencies)
    min_freq = min(frequencies)
    max_freq = max(frequencies)

    print(f"\nFrequency statistics:")
    print(f"  Mean: {mean_freq:.1f}")
    print(f"  Min:  {min_freq}")
    print(f"  Max:  {max_freq}")
    print(f"  Range: {max_freq - min_freq}")

    # Coefficient of variation (should be small for uniform)
    import math
    std_dev = math.sqrt(sum((f - mean_freq)**2 for f in frequencies) / len(frequencies))
    cv = std_dev / mean_freq if mean_freq > 0 else float('inf')

    print(f"  Std Dev: {std_dev:.2f}")
    print(f"  Coeff. of Variation: {cv:.3f}")

    # Show top 5 most and least common
    sorted_counts = sorted(subgraph_counts.items(), key=lambda x: x[1], reverse=True)

    print(f"\nMost common subgraphs:")
    for i, (sg, count) in enumerate(sorted_counts[:5]):
        print(f"  #{i+1}: {count} occurrences ({100*count/num_valid_samples:.1f}%)")

    print(f"\nLeast common subgraphs:")
    for i, (sg, count) in enumerate(sorted_counts[-5:]):
        print(f"  #{i+1}: {count} occurrences ({100*count/num_valid_samples:.1f}%)")

    # Uniformity assessment
    print("\n" + "=" * 60)
    print("UNIFORMITY ASSESSMENT")
    print("=" * 60)

    # For perfect uniformity, each should appear ~num_valid_samples/num_unique times
    expected_freq = num_valid_samples / num_unique
    max_deviation = max(abs(f - expected_freq) / expected_freq for f in frequencies) * 100

    print(f"Expected frequency (perfect uniform): {expected_freq:.1f}")
    print(f"Max deviation from expected: {max_deviation:.1f}%")

    if incomplete_samples > num_samples * 0.1:
        print("\n⚠️  WARNING: >10% incomplete samples - graph may be too sparse for k")
        print("   Uniformity guarantees may be broken (relaxations triggered)")
    elif cv < 0.15:
        print("\n✓ GOOD: Low variation - sampling appears roughly uniform")
    elif cv < 0.30:
        print("\n⚠️  MODERATE: Some variation - may indicate non-uniformity")
    else:
        print("\n✗ POOR: High variation - sampling is NOT uniform")
        print("   Relaxations may be triggering. Run with UGS_DEBUG=1 to check")

else:
    print("\n✗ ERROR: No valid subgraphs sampled!")
    print("Graph may be too sparse or disconnected for k={k}")

print("=" * 60)
