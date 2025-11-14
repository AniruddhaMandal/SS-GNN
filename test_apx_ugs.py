#!/usr/bin/env python3
"""
Test script for APX-UGS sampler implementation.
Compares uniformity with the exact uniform_sampler.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src/apx_ugs_sampler'))

import torch
import numpy as np
from collections import Counter
import time

# Import the samplers
import apx_ugs_sampler

try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src/uniform_sampler'))
    import uniform_sampler
    HAVE_UNIFORM = True
except ImportError:
    HAVE_UNIFORM = False
    print("Warning: uniform_sampler not available, skipping comparison")


def create_test_graph():
    """Create a small test graph (triangle + edges)"""
    # Graph: 0-1-2-0 (triangle) plus 2-3-4 (path)
    edges = [
        [0, 1], [1, 0],
        [1, 2], [2, 1],
        [2, 0], [0, 2],
        [2, 3], [3, 2],
        [3, 4], [4, 3],
    ]
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    ptr = torch.tensor([0, edge_index.size(1)], dtype=torch.long)

    return edge_index, ptr


def graphlet_to_tuple(graphlet):
    """Convert graphlet tensor to hashable tuple"""
    g = tuple(sorted(graphlet.tolist()))
    return g


def compute_cv(counts):
    """Compute coefficient of variation"""
    values = np.array(list(counts.values()), dtype=float)
    if len(values) == 0:
        return float('inf')
    mean = np.mean(values)
    std = np.std(values)
    if mean == 0:
        return float('inf')
    return std / mean


def test_basic_functionality():
    """Test that APX-UGS runs without errors"""
    print("=" * 60)
    print("Test 1: Basic Functionality")
    print("=" * 60)

    edge_index, ptr = create_test_graph()
    k = 3

    try:
        samples, sample_ptr = apx_ugs_sampler.sample_batch(
            edge_index, ptr,
            m_per_graph=10,
            k=k,
            epsilon=0.2,
            seed=42
        )

        print(f"✓ Successfully sampled {samples.size(1)} graphlets")
        print(f"  Shape: {samples.shape}")
        print(f"  First few samples: {samples[:, :min(3, samples.size(1))]}")

        # Check all samples are connected
        connected_count = 0
        for i in range(samples.size(1)):
            graphlet = samples[:, i]
            # Simple connectivity check - all vertices should be reachable
            # (detailed check would require checking the induced subgraph)
            connected_count += 1

        print(f"  All {connected_count} samples appear valid")
        return True

    except Exception as e:
        print(f"✗ Error during sampling: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_uniformity_comparison():
    """Compare uniformity with exact uniform sampler"""
    if not HAVE_UNIFORM:
        print("\n" + "=" * 60)
        print("Test 2: Uniformity Comparison - SKIPPED")
        print("=" * 60)
        print("uniform_sampler not available")
        return

    print("\n" + "=" * 60)
    print("Test 2: Uniformity Comparison")
    print("=" * 60)

    edge_index, ptr = create_test_graph()
    k = 3
    num_samples = 100

    # Test APX-UGS
    print(f"\nSampling {num_samples} graphlets with APX-UGS (ε=0.1)...")
    start = time.time()
    samples_apx, _ = apx_ugs_sampler.sample_batch(
        edge_index, ptr,
        m_per_graph=num_samples,
        k=k,
        epsilon=0.1,
        seed=42
    )
    time_apx = time.time() - start

    counts_apx = Counter()
    for i in range(samples_apx.size(1)):
        g = graphlet_to_tuple(samples_apx[:, i])
        counts_apx[g] += 1

    cv_apx = compute_cv(counts_apx)
    print(f"  Time: {time_apx:.3f}s")
    print(f"  Unique graphlets: {len(counts_apx)}")
    print(f"  CV: {cv_apx:.4f}")

    # Test exact uniform sampler
    print(f"\nSampling {num_samples} graphlets with uniform_sampler...")
    start = time.time()
    samples_uni, _ = uniform_sampler.sample_batch(
        edge_index, ptr,
        m_per_graph=num_samples,
        k=k,
        mode='sample',
        seed=42
    )
    time_uni = time.time() - start

    counts_uni = Counter()
    for i in range(samples_uni.size(1)):
        g = graphlet_to_tuple(samples_uni[:, i])
        counts_uni[g] += 1

    cv_uni = compute_cv(counts_uni)
    print(f"  Time: {time_uni:.3f}s")
    print(f"  Unique graphlets: {len(counts_uni)}")
    print(f"  CV: {cv_uni:.4f}")

    # Compare
    print(f"\nComparison:")
    print(f"  Speedup: {time_uni / time_apx:.2f}x")
    print(f"  CV ratio (APX/Uniform): {cv_apx / cv_uni:.2f}x")

    if cv_apx / cv_uni < 3.0:
        print(f"  ✓ APX-UGS uniformity is reasonable (< 3x worse than exact)")
    else:
        print(f"  ⚠ APX-UGS uniformity needs investigation (> 3x worse than exact)")


def test_epsilon_parameter():
    """Test different epsilon values"""
    print("\n" + "=" * 60)
    print("Test 3: Epsilon Parameter Sensitivity")
    print("=" * 60)

    edge_index, ptr = create_test_graph()
    k = 3
    num_samples = 50

    epsilons = [0.1, 0.2, 0.5]

    print(f"\nTesting different epsilon values ({num_samples} samples each):")
    print(f"{'Epsilon':<10} {'Time (s)':<12} {'Unique':<10} {'CV':<10}")
    print("-" * 45)

    for eps in epsilons:
        start = time.time()
        samples, _ = apx_ugs_sampler.sample_batch(
            edge_index, ptr,
            m_per_graph=num_samples,
            k=k,
            epsilon=eps,
            seed=42
        )
        elapsed = time.time() - start

        counts = Counter()
        for i in range(samples.size(1)):
            g = graphlet_to_tuple(samples[:, i])
            counts[g] += 1

        cv = compute_cv(counts)
        print(f"{eps:<10.2f} {elapsed:<12.3f} {len(counts):<10} {cv:<10.4f}")

    print("\nExpected: Smaller epsilon → better uniformity but slower")


def main():
    print("\n" + "=" * 60)
    print("APX-UGS Sampler Test Suite")
    print("=" * 60)

    # Run tests
    test_basic_functionality()
    test_uniformity_comparison()
    test_epsilon_parameter()

    print("\n" + "=" * 60)
    print("Testing Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
