#!/usr/bin/env python3
"""Quick test for epsilon_uniform_sampler"""

import sys
sys.path.insert(0, 'src/epsilon_uniform_sampler')

import torch
import epsilon_uniform_sampler

# Create a simple test graph: triangle + extra node
# Nodes: 0, 1, 2, 3
# Edges: (0-1), (1-2), (0-2), (2-3)
edge_index = torch.tensor([
    [0, 1, 1, 2, 0, 2, 2, 3],
    [1, 0, 2, 1, 2, 0, 3, 2]
], dtype=torch.int64)

# Single graph
ptr = torch.tensor([0, 4], dtype=torch.int64)

print("Testing epsilon_uniform_sampler")
print("=" * 50)
print(f"Graph: {edge_index.shape[1]//2} edges, 4 nodes")
print(f"Edge index:\n{edge_index}")
print()

# Test with different epsilon values
for epsilon in [0.01, 0.1, 0.5]:
    print(f"\nTesting with epsilon={epsilon}")
    print("-" * 40)

    try:
        nodes_t, edge_index_t, edge_ptr_t, sample_ptr_t, edge_src_t = \
            epsilon_uniform_sampler.sample_batch(
                edge_index,
                ptr,
                m_per_graph=5,  # 5 samples
                k=3,  # size 3 subgraphs
                mode="sample",
                seed=42,
                epsilon=epsilon
            )

        print(f"✓ Success!")
        print(f"  Nodes tensor shape: {nodes_t.shape}")
        print(f"  Sampled nodes (first 3 samples):")
        for i in range(min(3, nodes_t.shape[0])):
            print(f"    Sample {i}: {nodes_t[i].tolist()}")
        print(f"  Edge tensor shape: {edge_index_t.shape}")
        print(f"  Total edges sampled: {edge_index_t.shape[1]}")

    except Exception as e:
        print(f"✗ Failed: {e}")

print("\n" + "=" * 50)
print("Testing GPU support (if available)")
if torch.cuda.is_available():
    edge_index_cuda = edge_index.cuda()
    ptr_cuda = ptr.cuda()

    try:
        nodes_t, edge_index_t, edge_ptr_t, sample_ptr_t, edge_src_t = \
            epsilon_uniform_sampler.sample_batch(
                edge_index_cuda,
                ptr_cuda,
                m_per_graph=3,
                k=3,
                epsilon=0.1
            )
        print(f"✓ GPU test passed!")
        print(f"  Output device: {nodes_t.device}")
    except Exception as e:
        print(f"✗ GPU test failed: {e}")
else:
    print("CUDA not available, skipping GPU test")

print("\nAll tests completed!")
