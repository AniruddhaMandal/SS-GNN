#!/usr/bin/env python3
"""Test uniform_sampler - truly uniform connected subgraph sampling"""
import torch
import uniform_sampler
from collections import Counter
import math

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

print("=" * 70)
print("Testing UNIFORM SAMPLER - Truly Uniform Connected Subgraphs")
print("=" * 70)
print()

# Test 1: Simple graph
print("Test 1: Simple benzene-like ring (n=6, k=3)")
print("-" * 70)
num_nodes = 6
edge_index = torch.tensor([
    [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0],
    [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 0, 5],
], dtype=torch.long)
ptr = torch.tensor([0, num_nodes], dtype=torch.long)

k = 3
num_samples = 1000

nodes_t, edge_index_t, edge_ptr_t, sample_ptr_t, edge_src_t = \
    uniform_sampler.sample_batch(edge_index, ptr, m_per_graph=num_samples, k=k, seed=42)

print(f"Output shapes:")
print(f"  nodes_t: {nodes_t.shape}")
print(f"  edge_index_t: {edge_index_t.shape}")
print(f"  edge_ptr_t: {edge_ptr_t.shape}")
print(f"  sample_ptr_t: {sample_ptr_t.shape}")
print(f"  edge_src_t: {edge_src_t.shape}")
print()

# Count unique subgraphs
subgraph_counts = Counter()

for i in range(num_samples):
    nodes = nodes_t[i]
    edge_start = edge_ptr_t[i].item()
    edge_end = edge_ptr_t[i+1].item()
    edges = edge_index_t[:, edge_start:edge_end]

    if (nodes >= 0).sum() < k:
        continue

    canonical = graph_to_canonical(nodes[nodes >= 0], edges)
    subgraph_counts[canonical] += 1

num_valid = sum(subgraph_counts.values())
num_unique = len(subgraph_counts)

print(f"Valid samples: {num_valid}/{num_samples}")
print(f"Unique subgraphs: {num_unique}")

if num_unique > 0:
    frequencies = list(subgraph_counts.values())
    mean_freq = sum(frequencies) / len(frequencies)
    std_dev = math.sqrt(sum((f - mean_freq)**2 for f in frequencies) / len(frequencies))
    cv = std_dev / mean_freq if mean_freq > 0 else float('inf')

    print(f"Frequency stats:")
    print(f"  Mean: {mean_freq:.1f}")
    print(f"  Std:  {std_dev:.2f}")
    print(f"  CV:   {cv:.3f}")

    if cv < 0.15:
        print(f"  ✓ EXCELLENT uniformity (CV < 0.15)")
    elif cv < 0.30:
        print(f"  ⚠️  MODERATE uniformity (CV < 0.30)")
    else:
        print(f"  ✗ POOR uniformity (CV >= 0.30)")

print()

# Test 2: QM9 molecule
print("Test 2: QM9 molecule (n=11, k=4)")
print("-" * 70)

try:
    from torch_geometric.datasets import QM9
    dataset = QM9(root='./data/QM9')
    data = dataset[100]

    edge_index = data.edge_index
    num_nodes = data.num_nodes
    ptr = torch.tensor([0, num_nodes], dtype=torch.long)

    k = 4
    num_samples = 5000

    print(f"Graph: {num_nodes} nodes, {edge_index.size(1)} edges")
    print(f"Sampling {num_samples} subgraphs of size k={k}")
    print()

    nodes_t, edge_index_t, edge_ptr_t, sample_ptr_t, edge_src_t = \
        uniform_sampler.sample_batch(edge_index, ptr, m_per_graph=num_samples, k=k, seed=42)

    # Count unique subgraphs
    subgraph_counts = Counter()

    for i in range(num_samples):
        nodes = nodes_t[i]
        edge_start = edge_ptr_t[i].item()
        edge_end = edge_ptr_t[i+1].item()
        edges = edge_index_t[:, edge_start:edge_end]

        if (nodes >= 0).sum() < k:
            continue

        canonical = graph_to_canonical(nodes[nodes >= 0], edges)
        subgraph_counts[canonical] += 1

    num_valid = sum(subgraph_counts.values())
    num_unique = len(subgraph_counts)

    print(f"Valid samples: {num_valid}/{num_samples}")
    print(f"Unique subgraphs found: {num_unique}")

    if num_unique > 0:
        frequencies = list(subgraph_counts.values())
        mean_freq = sum(frequencies) / len(frequencies)
        min_freq = min(frequencies)
        max_freq = max(frequencies)
        std_dev = math.sqrt(sum((f - mean_freq)**2 for f in frequencies) / len(frequencies))
        cv = std_dev / mean_freq if mean_freq > 0 else float('inf')

        print(f"\nFrequency statistics:")
        print(f"  Mean:   {mean_freq:.1f}")
        print(f"  Min:    {min_freq}")
        print(f"  Max:    {max_freq}")
        print(f"  Range:  {max_freq - min_freq}")
        print(f"  Std:    {std_dev:.2f}")
        print(f"  CV:     {cv:.3f}")

        # Show distribution
        sorted_counts = sorted(subgraph_counts.items(), key=lambda x: x[1], reverse=True)
        print(f"\nMost common (top 3):")
        for i, (sg, count) in enumerate(sorted_counts[:3]):
            print(f"  #{i+1}: {count} occurrences ({100*count/num_valid:.1f}%)")

        print(f"\nLeast common (bottom 3):")
        for i, (sg, count) in enumerate(sorted_counts[-3:]):
            print(f"  #{i+1}: {count} occurrences ({100*count/num_valid:.1f}%)")

        # Uniformity assessment
        print(f"\n" + "=" * 70)
        print("UNIFORMITY ASSESSMENT")
        print("=" * 70)

        expected_freq = num_valid / num_unique
        max_deviation = max(abs(f - expected_freq) / expected_freq for f in frequencies) * 100

        print(f"Expected frequency (perfect uniform): {expected_freq:.1f}")
        print(f"Max deviation from expected: {max_deviation:.1f}%")

        if cv < 0.15:
            print("\n✓ EXCELLENT: Very low variation - sampling is truly uniform!")
        elif cv < 0.30:
            print("\n⚠️  MODERATE: Some variation present")
        else:
            print("\n✗ POOR: High variation - NOT uniform")

except ImportError:
    print("PyTorch Geometric not available, skipping QM9 test")

print()

# Test 3: Seed reproducibility
print("Test 3: Seed reproducibility")
print("-" * 70)

num_nodes = 6
edge_index = torch.tensor([
    [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0],
    [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 0, 5],
], dtype=torch.long)
ptr = torch.tensor([0, num_nodes], dtype=torch.long)

# Sample twice with same seed
nodes1, _, _, _, _ = uniform_sampler.sample_batch(edge_index, ptr, m_per_graph=10, k=3, seed=12345)
nodes2, _, _, _, _ = uniform_sampler.sample_batch(edge_index, ptr, m_per_graph=10, k=3, seed=12345)

# Sample with different seed
nodes3, _, _, _, _ = uniform_sampler.sample_batch(edge_index, ptr, m_per_graph=10, k=3, seed=54321)

same_seed_match = torch.equal(nodes1, nodes2)
diff_seed_match = torch.equal(nodes1, nodes3)

print(f"Same seed produces same results: {same_seed_match} {'✓' if same_seed_match else '✗'}")
print(f"Different seed produces different results: {not diff_seed_match} {'✓' if not diff_seed_match else '✗'}")

print()
print("=" * 70)
print("ALL TESTS COMPLETED")
print("=" * 70)
