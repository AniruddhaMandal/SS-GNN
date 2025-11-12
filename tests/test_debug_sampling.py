#!/usr/bin/env python3
"""Debug: Trace a few samples to understand non-uniformity"""
import torch
import ugs_sampler

# Small graph for easier analysis
edge_index = torch.tensor([
    [0, 1, 1, 2, 2, 0, 2, 3],  # Triangle 0-1-2 + tail 3
    [1, 0, 2, 1, 0, 2, 3, 2],
], dtype=torch.long)
num_nodes = 4
k = 3

print("Graph: Triangle 0-1-2 with tail 3")
print("All possible 3-node connected subgraphs:")
print("  1. {0,1,2} - triangle")
print("  2. {1,2,3} - path")
print("  3. {0,1,3} - NOT connected")
print("  4. {0,2,3} - path")
print("\nSo there are 3 connected 3-subgraphs")
print("Uniform sampling: each should appear ~33.3% of the time\n")

# Sample many times
num_samples = 3000
samples_count = {}

for i in range(num_samples):
    handle = ugs_sampler.create_preproc(edge_index, num_nodes, k)
    nodes_t, _, _, _ = ugs_sampler.sample(handle, 1, k, "local", 0)
    ugs_sampler.destroy_preproc(handle)

    # Get canonical representation
    nodes = tuple(sorted([nodes_t[0, j].item() for j in range(k) if nodes_t[0, j] >= 0]))
    samples_count[nodes] = samples_count.get(nodes, 0) + 1

print(f"Sampling results ({num_samples} samples):")
total = sum(samples_count.values())
for nodes, count in sorted(samples_count.items(), key=lambda x: -x[1]):
    pct = 100.0 * count / total
    expected = 100.0 / len(samples_count)
    print(f"  {nodes}: {count}/{total} ({pct:.1f}%) [expected: {expected:.1f}%]")

print(f"\n{'✓ UNIFORM' if max(samples_count.values()) / min(samples_count.values()) < 1.5 else '✗ NOT UNIFORM'}")
