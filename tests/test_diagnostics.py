#!/usr/bin/env python3
"""Quick test to see UGS diagnostic messages

To enable diagnostic logging, run with:
    UGS_DEBUG=1 python tests/test_diagnostics.py

Diagnostics show:
- [UGS INFO]: Preprocessing overhead warnings
- [UGS WARNING]: When uniformity guarantees are broken
- [UGS DIAGNOSTIC]: Incomplete samples and retry statistics
"""
import torch
import ugs_sampler
import os

print("=" * 60)
print("Testing UGS Sampler with Diagnostics")
if os.environ.get("UGS_DEBUG") == "1":
    print("(DEBUG MODE ENABLED)")
else:
    print("(Run with UGS_DEBUG=1 to see diagnostics)")
print("=" * 60)

# Test 1: Well-connected graph (should work fine)
print("\n[TEST 1] Well-connected graph (4 nodes, triangle + 1)")
ptr = torch.tensor([0, 4], dtype=torch.long)
edge_index = torch.tensor([
    [0, 1, 1, 2, 2, 0],  # triangle 0-1-2
    [1, 0, 2, 1, 0, 2],
], dtype=torch.long)

try:
    result = ugs_sampler.sample_batch(edge_index, ptr, m_per_graph=5, k=3, mode="sample")
    print(f"✓ Success: sampled {result[0].shape[0]} subgraphs")
except Exception as e:
    print(f"✗ Failed: {e}")

# Test 2: Sparse graph (may trigger warnings)
print("\n[TEST 2] Sparse graph (chain of 5 nodes)")
ptr = torch.tensor([0, 5], dtype=torch.long)
edge_index = torch.tensor([
    [0, 1, 2, 3],  # chain: 0-1-2-3-4
    [1, 2, 3, 4],
], dtype=torch.long)

try:
    result = ugs_sampler.sample_batch(edge_index, ptr, m_per_graph=10, k=4, mode="sample")
    print(f"✓ Success: sampled {result[0].shape[0]} subgraphs")
except Exception as e:
    print(f"✗ Failed: {e}")

# Test 3: Very sparse (likely to trigger warnings)
print("\n[TEST 3] Very sparse graph (3 nodes, single edge)")
ptr = torch.tensor([0, 3], dtype=torch.long)
edge_index = torch.tensor([
    [0, 1],
    [1, 0],
], dtype=torch.long)

try:
    result = ugs_sampler.sample_batch(edge_index, ptr, m_per_graph=5, k=3, mode="sample")
    print(f"✓ Success: sampled {result[0].shape[0]} subgraphs")
except Exception as e:
    print(f"✗ Failed: {e}")

print("\n" + "=" * 60)
print("Look above for [UGS WARNING] or [UGS DIAGNOSTIC] messages")
print("=" * 60)
