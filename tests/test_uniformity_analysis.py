#!/usr/bin/env python3
"""Deep analysis of sampling uniformity"""

import sys
sys.path.insert(0, 'src/epsilon_uniform_sampler')
sys.path.insert(0, 'src/uniform_sampler')

import torch
import numpy as np
from collections import Counter
import epsilon_uniform_sampler
import uniform_sampler

def create_simple_graph():
    """Create a simple well-understood graph"""
    # Triangle graph: nodes 0,1,2 all connected
    edges = [
        [0, 1], [1, 0],
        [1, 2], [2, 1],
        [0, 2], [2, 0],
    ]
    edge_index = torch.tensor(edges, dtype=torch.int64).t()
    ptr = torch.tensor([0, 3], dtype=torch.int64)
    return edge_index, ptr

def theoretical_cv(n_samples, n_categories):
    """Calculate theoretical CV for uniform multinomial sampling"""
    # Each category follows Binomial(n, p) where p = 1/n_categories
    p = 1.0 / n_categories
    expected_count = n_samples * p
    variance = n_samples * p * (1 - p)
    std_dev = np.sqrt(variance)
    cv = std_dev / expected_count
    return cv, expected_count, std_dev

def chi_square_test(observed_counts, expected_count):
    """Chi-square goodness of fit test"""
    chi_square = sum((count - expected_count)**2 / expected_count
                     for count in observed_counts)
    df = len(observed_counts) - 1
    # Critical value at 0.05 significance for various df
    critical_values = {5: 11.07, 10: 18.31, 15: 25.00, 20: 31.41, 25: 37.65}
    cv_approx = critical_values.get(min(critical_values.keys(),
                                         key=lambda x: abs(x - df)), 31.41)
    return chi_square, df, chi_square < cv_approx

print("=" * 70)
print("UNIFORMITY ANALYSIS: Investigating the CV=0.54 Mystery")
print("=" * 70)

# Test 1: Simple triangle graph where we know the answer
print("\nTest 1: Triangle Graph (Sanity Check)")
print("-" * 70)
edge_index, ptr = create_simple_graph()
print("Graph: Triangle (nodes 0,1,2 all connected)")
print("Expected: Only 1 possible connected 3-subgraph: {0,1,2}")

# Test uniform sampler
print("\n[Uniform Sampler Test]")
samples = []
for i in range(10):
    nodes_t, _, _, _, _ = uniform_sampler.sample_batch(
        edge_index, ptr, 1, k=3, seed=i
    )
    samples.append(tuple(sorted(nodes_t[0].tolist())))

unique = set(samples)
print(f"Unique subgraphs found: {unique}")
print(f"✓ PASS" if unique == {(0,1,2)} else "✗ FAIL - unexpected subgraphs!")

# Test 2: Theoretical CV analysis
print("\n" + "=" * 70)
print("Test 2: Theoretical vs Observed CV")
print("-" * 70)

n_samples = 1000
for n_categories in [10, 16, 20, 30]:
    theoretical_cv_val, exp_count, std = theoretical_cv(n_samples, n_categories)
    print(f"\nCategories: {n_categories}, Samples: {n_samples}")
    print(f"  Theoretical CV: {theoretical_cv_val:.4f}")
    print(f"  Expected count per category: {exp_count:.1f}")
    print(f"  Expected std dev: {std:.2f}")
    print(f"  Expected range (±2σ): [{exp_count-2*std:.1f}, {exp_count+2*std:.1f}]")

print(f"\n⚠ Observed CV of 0.54 is ~4.5x higher than theoretical 0.12!")
print(f"   This suggests the sampling may NOT be perfectly uniform.")

# Test 3: Deeper investigation with known graph
print("\n" + "=" * 70)
print("Test 3: Controlled Uniformity Test")
print("-" * 70)

# Create a simple graph where we can enumerate all subgraphs
# 4-node graph: 0-1-2-3 (path) with 0-2 edge (makes triangle 0-1-2)
edges = [
    [0, 1], [1, 0],
    [1, 2], [2, 1],
    [2, 3], [3, 2],
    [0, 2], [2, 0],  # Makes triangle
]
edge_index = torch.tensor(edges, dtype=torch.int64).t()
ptr = torch.tensor([0, 4], dtype=torch.int64)

print("Graph: 0-1-2-3 path with 0-2 edge (triangle 0-1-2)")
print("Connected 3-subgraphs should be:")
print("  {0,1,2} - triangle")
print("  {1,2,3} - path")

# First, verify what the uniform sampler actually finds
print("\n[Enumerating via Uniform Sampler]")
all_found = set()
for seed in range(100):
    nodes_t, _, _, _, _ = uniform_sampler.sample_batch(
        edge_index, ptr, 1, k=3, seed=seed
    )
    sg = tuple(sorted(nodes_t[0].tolist()))
    all_found.add(sg)

print(f"Unique subgraphs found: {sorted(all_found)}")
print(f"Count: {len(all_found)}")

# Now test uniformity with large sample
print("\n[Large-Scale Uniformity Test: 10,000 samples]")
n_test_samples = 10000

# Uniform sampler
uniform_counts = Counter()
for i in range(n_test_samples):
    nodes_t, _, _, _, _ = uniform_sampler.sample_batch(
        edge_index, ptr, 1, k=3, seed=i
    )
    sg = tuple(sorted(nodes_t[0].tolist()))
    uniform_counts[sg] += 1

print("\nUniform Sampler Results:")
for sg, count in sorted(uniform_counts.items()):
    print(f"  {sg}: {count} times ({100*count/n_test_samples:.1f}%)")

# Calculate statistics
values = list(uniform_counts.values())
mean_count = np.mean(values)
std_count = np.std(values, ddof=1)
cv_observed = std_count / mean_count
min_count, max_count = min(values), max(values)

print(f"\nStatistics:")
print(f"  Mean: {mean_count:.1f}")
print(f"  Std Dev: {std_count:.2f}")
print(f"  CV (observed): {cv_observed:.4f}")
print(f"  Range: [{min_count}, {max_count}]")

# Compare with theoretical
n_cat = len(uniform_counts)
theo_cv, theo_mean, theo_std = theoretical_cv(n_test_samples, n_cat)
print(f"\nTheoretical (if perfectly uniform):")
print(f"  Expected CV: {theo_cv:.4f}")
print(f"  Expected mean: {theo_mean:.1f}")
print(f"  Expected std: {theo_std:.2f}")
print(f"  Expected range (±2σ): [{theo_mean-2*theo_std:.1f}, {theo_mean+2*theo_std:.1f}]")

# Chi-square test
chi2, df, is_uniform = chi_square_test(values, theo_mean)
print(f"\nChi-Square Test:")
print(f"  χ² = {chi2:.2f}, df = {df}")
print(f"  Result: {'✓ PASS - Distribution is uniform' if is_uniform else '✗ FAIL - Distribution is NOT uniform'}")

# Test epsilon sampler for comparison
print("\n[Epsilon-Uniform Sampler (ε=0.1)]")
epsilon_counts = Counter()
for i in range(n_test_samples):
    nodes_t, _, _, _, _ = epsilon_uniform_sampler.sample_batch(
        edge_index, ptr, 1, k=3, seed=i, epsilon=0.1
    )
    sg = tuple(sorted(nodes_t[0].tolist()))
    epsilon_counts[sg] += 1

print("\nEpsilon-Uniform Results:")
for sg, count in sorted(epsilon_counts.items()):
    print(f"  {sg}: {count} times ({100*count/n_test_samples:.1f}%)")

eps_values = list(epsilon_counts.values())
eps_mean = np.mean(eps_values)
eps_std = np.std(eps_values, ddof=1)
eps_cv = eps_std / eps_mean

print(f"\nStatistics:")
print(f"  CV (observed): {eps_cv:.4f}")
print(f"  Comparison: {eps_cv/cv_observed:.2f}x worse than 'uniform' sampler")

print("\n" + "=" * 70)
print("CONCLUSIONS:")
print("-" * 70)
print(f"1. Theoretical CV for uniform sampling: ~{theo_cv:.3f}")
print(f"2. Observed CV for 'uniform' sampler: {cv_observed:.3f}")
print(f"3. Observed CV for epsilon sampler: {eps_cv:.3f}")
print()
if cv_observed > theo_cv * 2:
    print("⚠ WARNING: 'Uniform' sampler CV is significantly higher than expected!")
    print("   This could indicate:")
    print("   - Statistical fluctuation (unlikely with 10k samples)")
    print("   - Graph structure effects")
    print("   - Potential implementation issue")
else:
    print("✓ 'Uniform' sampler CV is within expected range")

print()
ratio = eps_cv / cv_observed
print(f"Epsilon sampler is {ratio:.2f}x less uniform than 'exact' sampler")
if ratio < 2.0:
    print(f"✓ This is ACCEPTABLE for practical use (massive speedup for small cost)")
else:
    print(f"⚠ This may be TOO much deviation for some applications")

print("=" * 70)
