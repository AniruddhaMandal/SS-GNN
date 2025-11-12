#!/usr/bin/env python3
"""Examine bucket weights to understand non-uniformity"""
import torch
import ugs_sampler

try:
    from torch_geometric.datasets import QM9
    dataset = QM9(root='/tmp/QM9')
    data = dataset[100]
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    print(f"QM9 molecule 100: {num_nodes} nodes, {edge_index.size(1)} edges")
except:
    # Fallback to synthetic
    num_nodes = 6
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0],
        [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 0, 5],
    ], dtype=torch.long)
    print(f"Synthetic graph: {num_nodes} nodes")

k = 4

# Create preprocessing
handle = ugs_sampler.create_preproc(edge_index, num_nodes, k)

# Get preprocessing info
info = ugs_sampler.get_preproc_info(handle)

print(f"\nPreprocessing info:")
print(f"  Z (total weight): {info['Z']:.2e}")
print(f"  Viable roots (b>0): {info['bucket_count_nonzero']}/{info['num_nodes']}")

# Sample and analyze root distribution
num_samples = 10000
root_counts = {}

for i in range(num_samples):
    # Sample using the handle
    nodes_t, _, _, _ = ugs_sampler.sample(handle, 1, k, "local", 0)
    root = nodes_t[0, 0].item()
    root_counts[root] = root_counts.get(root, 0) + 1

print(f"\nRoot selection distribution ({num_samples} samples):")
print("(If weighted sampling works correctly, roots with higher bucket weights should be selected more often)")

# Sort by frequency
sorted_roots = sorted(root_counts.items(), key=lambda x: x[1], reverse=True)
for i, (root, count) in enumerate(sorted_roots[:10]):
    pct = 100.0 * count / num_samples
    print(f"  Root {root}: {count} times ({pct:.1f}%)")

# Cleanup
ugs_sampler.destroy_preproc(handle)
