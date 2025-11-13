#!/usr/bin/env python3
"""Compare uniform_sampler vs UGS sampler on QM9"""
import torch
import uniform_sampler
import ugs_sampler
from collections import Counter
import math

def graph_to_canonical(nodes, edge_index):
    """Convert sampled subgraph to canonical form for counting"""
    nodes_sorted = tuple(sorted(nodes.tolist()))
    edges = []
    node_map = {old: new for new, old in enumerate(nodes_sorted)}

    for i in range(edge_index.size(1)):
        u, v = edge_index[0, i].item(), edge_index[1, i].item()
        if u in node_map and v in node_map:
            u_new, v_new = node_map[u], node_map[v]
            edges.append(tuple(sorted([u_new, v_new])))

    return (nodes_sorted, tuple(sorted(set(edges))))

try:
    from torch_geometric.datasets import QM9
    HAS_QM9 = True
except ImportError:
    HAS_QM9 = False
    print("PyTorch Geometric not available")
    exit(1)

print("=" * 80)
print("COMPARISON: Uniform Sampler vs UGS Sampler")
print("=" * 80)
print()

# Load QM9 graph
dataset = QM9(root='./data/QM9')
data = dataset[100]

edge_index = data.edge_index
num_nodes = data.num_nodes
ptr = torch.tensor([0, num_nodes], dtype=torch.long)

k = 4
num_samples = 5000

print(f"Test Graph: QM9 molecule #100")
print(f"  Nodes: {num_nodes}")
print(f"  Edges: {edge_index.size(1)}")
print(f"  Subgraph size k: {k}")
print(f"  Number of samples: {num_samples}")
print()

# Test 1: Uniform Sampler
print("-" * 80)
print("Test 1: UNIFORM SAMPLER (Exhaustive Enumeration)")
print("-" * 80)

nodes_t, edge_index_t, edge_ptr_t, sample_ptr_t, edge_src_t = \
    uniform_sampler.sample_batch(edge_index, ptr, m_per_graph=num_samples, k=k, seed=42)

uniform_counts = Counter()
for i in range(num_samples):
    nodes = nodes_t[i]
    edge_start = edge_ptr_t[i].item()
    edge_end = edge_ptr_t[i+1].item()
    edges = edge_index_t[:, edge_start:edge_end]

    if (nodes >= 0).sum() < k:
        continue

    canonical = graph_to_canonical(nodes[nodes >= 0], edges)
    uniform_counts[canonical] += 1

num_valid_uniform = sum(uniform_counts.values())
num_unique_uniform = len(uniform_counts)

frequencies_uniform = list(uniform_counts.values())
mean_uniform = sum(frequencies_uniform) / len(frequencies_uniform)
std_uniform = math.sqrt(sum((f - mean_uniform)**2 for f in frequencies_uniform) / len(frequencies_uniform))
cv_uniform = std_uniform / mean_uniform
min_uniform = min(frequencies_uniform)
max_uniform = max(frequencies_uniform)

print(f"Results:")
print(f"  Valid samples: {num_valid_uniform}/{num_samples}")
print(f"  Unique subgraphs: {num_unique_uniform}")
print(f"  Frequency range: [{min_uniform}, {max_uniform}]")
print(f"  Mean frequency: {mean_uniform:.1f}")
print(f"  Std deviation: {std_uniform:.2f}")
print(f"  Coeff. of Variation: {cv_uniform:.3f}")
print(f"  Max/Min ratio: {max_uniform/min_uniform:.2f}x")
print()

# Test 2: UGS Sampler
print("-" * 80)
print("Test 2: UGS SAMPLER (Bressan's Approximate)")
print("-" * 80)

nodes_t, edge_index_t, edge_ptr_t, sample_ptr_t, edge_src_t = \
    ugs_sampler.sample_batch(edge_index, ptr, m_per_graph=num_samples, k=k, mode="sample")

ugs_counts = Counter()
for i in range(num_samples):
    nodes = nodes_t[i]
    edge_start = edge_ptr_t[i].item()
    edge_end = edge_ptr_t[i+1].item()
    edges = edge_index_t[:, edge_start:edge_end]

    if (nodes >= 0).sum() < k:
        continue

    canonical = graph_to_canonical(nodes[nodes >= 0], edges)
    ugs_counts[canonical] += 1

num_valid_ugs = sum(ugs_counts.values())
num_unique_ugs = len(ugs_counts)

frequencies_ugs = list(ugs_counts.values())
mean_ugs = sum(frequencies_ugs) / len(frequencies_ugs)
std_ugs = math.sqrt(sum((f - mean_ugs)**2 for f in frequencies_ugs) / len(frequencies_ugs))
cv_ugs = std_ugs / mean_ugs
min_ugs = min(frequencies_ugs)
max_ugs = max(frequencies_ugs)

print(f"Results:")
print(f"  Valid samples: {num_valid_ugs}/{num_samples}")
print(f"  Unique subgraphs: {num_unique_ugs}")
print(f"  Frequency range: [{min_ugs}, {max_ugs}]")
print(f"  Mean frequency: {mean_ugs:.1f}")
print(f"  Std deviation: {std_ugs:.2f}")
print(f"  Coeff. of Variation: {cv_ugs:.3f}")
print(f"  Max/Min ratio: {max_ugs/min_ugs:.2f}x")
print()

# Comparison
print("=" * 80)
print("COMPARISON SUMMARY")
print("=" * 80)
print()

print(f"{'Metric':<30} {'Uniform Sampler':<20} {'UGS Sampler':<20} {'Improvement':<15}")
print("-" * 80)
print(f"{'Coeff. of Variation':<30} {cv_uniform:<20.3f} {cv_ugs:<20.3f} {cv_ugs/cv_uniform:.1f}x worse")
print(f"{'Max/Min Ratio':<30} {max_uniform/min_uniform:<20.2f} {max_ugs/min_ugs:<20.2f} {(max_ugs/min_ugs)/(max_uniform/min_uniform):.1f}x worse")
print(f"{'Std Deviation':<30} {std_uniform:<20.2f} {std_ugs:<20.2f} {std_ugs/std_uniform:.1f}x worse")
print()

print("Conclusion:")
if cv_uniform < 0.15:
    print("  ✓ Uniform Sampler: EXCELLENT uniformity (CV < 0.15)")
else:
    print("  ⚠️  Uniform Sampler: MODERATE uniformity")

if cv_ugs > 0.30:
    print("  ✗ UGS Sampler: POOR uniformity (CV > 0.30) - significant bias present")
elif cv_ugs > 0.15:
    print("  ⚠️  UGS Sampler: MODERATE uniformity (CV > 0.15) - some bias present")
else:
    print("  ✓ UGS Sampler: GOOD uniformity")

print()
print(f"Uniformity improvement: {cv_ugs/cv_uniform:.1f}x better with Uniform Sampler")
print()
print("=" * 80)
