#!/usr/bin/env python3
"""
Measure subgraph distribution divergence for a dataset.
Run this BEFORE training to check if SS-GNN will work well.

Usage:
    python measure_subgraph_divergence.py --dataset CSL --k 4 --num_samples 100
    python measure_subgraph_divergence.py --dataset MUTAG --k 5 --num_samples 50
"""

import argparse
import numpy as np
import networkx as nx
from collections import Counter
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
import random
from tqdm import tqdm


def sample_k_subgraphs(edge_index, num_nodes, k, num_samples, method='random_walk'):
    """
    Sample k-node connected subgraphs from a graph.

    Args:
        edge_index: [2, E] tensor or list of edges
        num_nodes: number of nodes in graph
        k: subgraph size
        num_samples: number of subgraphs to sample
        method: 'random_walk' or 'uniform'

    Returns:
        List of subgraph signatures (canonical form)
    """
    # Convert to networkx
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))

    if hasattr(edge_index, 'numpy'):
        edges = edge_index.t().cpu().numpy().tolist()
    elif hasattr(edge_index, 'tolist'):
        edges = edge_index.tolist()
        if len(edges) == 2 and isinstance(edges[0], list):  # [2, E] format
            edges = list(zip(edges[0], edges[1]))
    else:
        edges = edge_index

    G.add_edges_from(edges)

    # Sample subgraphs
    subgraphs = []
    max_attempts = num_samples * 10
    attempts = 0

    while len(subgraphs) < num_samples and attempts < max_attempts:
        attempts += 1

        if method == 'random_walk':
            # Random walk to get connected subgraph
            if num_nodes < k:
                continue

            start = random.randint(0, num_nodes - 1)
            nodes = {start}
            current = start

            # Random walk until we have k nodes
            for _ in range(k * 3):  # More iterations for better coverage
                if len(nodes) >= k:
                    break
                neighbors = list(G.neighbors(current))
                if not neighbors:
                    break
                next_node = random.choice(neighbors)
                nodes.add(next_node)
                current = next_node

            if len(nodes) < k:
                continue

            # Take first k nodes
            nodes = list(nodes)[:k]

        else:  # uniform (slower, more accurate)
            # Rejection sampling
            nodes = random.sample(range(num_nodes), min(k, num_nodes))
            # Check if connected
            subgraph = G.subgraph(nodes)
            if not nx.is_connected(subgraph):
                continue

        # Get canonical signature
        subgraph = G.subgraph(nodes)

        # Signature: (sorted degree sequence, num edges)
        degrees = sorted([d for n, d in subgraph.degree()])
        num_edges = subgraph.number_of_edges()
        signature = (tuple(degrees), num_edges)

        subgraphs.append(signature)

    return subgraphs


def compute_distribution(subgraph_list):
    """Convert list of subgraphs to probability distribution."""
    counter = Counter(subgraph_list)
    total = sum(counter.values())

    # Get all unique types
    types = sorted(counter.keys())

    # Probability vector
    probs = np.array([counter[t] / total for t in types])

    return types, probs, counter


def align_distributions(types1, probs1, types2, probs2):
    """Align two distributions to have same support (for JS divergence)."""
    # Union of types
    all_types = sorted(set(types1) | set(types2))

    # Create aligned probability vectors
    prob_vec1 = np.zeros(len(all_types))
    prob_vec2 = np.zeros(len(all_types))

    type_to_idx = {t: i for i, t in enumerate(all_types)}

    for t, p in zip(types1, probs1):
        prob_vec1[type_to_idx[t]] = p

    for t, p in zip(types2, probs2):
        prob_vec2[type_to_idx[t]] = p

    return prob_vec1, prob_vec2, all_types


def measure_divergence(dataset_name, k, num_samples=100, method='random_walk'):
    """
    Measure JS divergence between classes in a dataset.

    Args:
        dataset_name: Name of dataset ('CSL', 'MUTAG', etc.)
        k: Subgraph size
        num_samples: Number of subgraphs to sample per graph
        method: Sampling method

    Returns:
        Dictionary with divergence statistics
    """
    print("=" * 80)
    print(f"SUBGRAPH DISTRIBUTION DIVERGENCE ANALYSIS")
    print("=" * 80)
    print(f"\nDataset: {dataset_name}")
    print(f"Subgraph size (k): {k}")
    print(f"Samples per graph (m): {num_samples}")
    print(f"Sampling method: {method}")

    # Load dataset
    print("\nLoading dataset...")

    try:
        # Try loading from torch_geometric
        from torch_geometric.datasets import TUDataset, GNNBenchmarkDataset

        if dataset_name == 'CSL':
            dataset = GNNBenchmarkDataset(root='data', name='CSL')
        else:
            dataset = TUDataset(root='data/TUDataset', name=dataset_name)
    except:
        # Try loading from synthetic dataset
        from synthetic_dataset import SyntheticGraphData
        syn_data = SyntheticGraphData(cache_dir='./data/SYNTHETIC-DATA')
        dataset = syn_data.get(dataset_name, cache=False)

    print(f"Loaded {len(dataset)} graphs")

    # Group by class
    graphs_by_class = {}
    for data in dataset:
        label = int(data.y.item())
        if label not in graphs_by_class:
            graphs_by_class[label] = []
        graphs_by_class[label].append(data)

    num_classes = len(graphs_by_class)
    print(f"Number of classes: {num_classes}")

    for cls, graphs in graphs_by_class.items():
        print(f"  Class {cls}: {len(graphs)} graphs")

    # Sample subgraphs for each class
    print("\nSampling subgraphs from each class...")

    class_distributions = {}

    for cls in tqdm(sorted(graphs_by_class.keys()), desc="Classes"):
        # Take first graph from class (or sample from multiple)
        graph = graphs_by_class[cls][0]

        # Sample subgraphs
        subgraphs = sample_k_subgraphs(
            graph.edge_index,
            graph.num_nodes,
            k,
            num_samples,
            method=method
        )

        # Compute distribution
        types, probs, counter = compute_distribution(subgraphs)
        class_distributions[cls] = (types, probs, counter)

        # Print statistics
        print(f"\n  Class {cls}:")
        print(f"    Sampled: {len(subgraphs)} subgraphs")
        print(f"    Unique types: {len(types)}")
        print(f"    Top 3 types:")
        for sig, count in counter.most_common(3):
            degrees, edges = sig
            print(f"      Degrees: {degrees}, Edges: {edges}, Count: {count}")

    # Compute pairwise JS divergences
    print("\n" + "=" * 80)
    print("JENSEN-SHANNON DIVERGENCE ANALYSIS")
    print("=" * 80)

    divergences = []
    divergence_matrix = np.zeros((num_classes, num_classes))

    class_ids = sorted(graphs_by_class.keys())

    for i, cls1 in enumerate(class_ids):
        for j, cls2 in enumerate(class_ids):
            if i >= j:
                continue

            types1, probs1, _ = class_distributions[cls1]
            types2, probs2, _ = class_distributions[cls2]

            # Align distributions
            p1, p2, _ = align_distributions(types1, probs1, types2, probs2)

            # Compute JS divergence
            js_div = jensenshannon(p1, p2)

            divergences.append(js_div)
            divergence_matrix[i, j] = js_div
            divergence_matrix[j, i] = js_div

            if i < 3 and j < 3:  # Show first few
                print(f"  Class {cls1} vs Class {cls2}: {js_div:.6f}")

    mean_div = np.mean(divergences)
    std_div = np.std(divergences)
    min_div = np.min(divergences)
    max_div = np.max(divergences)

    print(f"\nSummary Statistics:")
    print(f"  Mean divergence: {mean_div:.6f} ± {std_div:.6f}")
    print(f"  Min divergence:  {min_div:.6f}")
    print(f"  Max divergence:  {max_div:.6f}")

    # Recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATION FOR SS-GNN")
    print("=" * 80)

    if mean_div < 0.1:
        verdict = "❌ POOR"
        explanation = """
Subgraph distributions are nearly IDENTICAL across classes.
SS-GNN theory barely applies to this dataset with k={k}.

Recommendations:
  1. Try larger k (k={k+2}, k={k+4})
  2. This dataset may not have discriminative k-subgraph structure
  3. Consider using vanilla GNN or other methods

Expected performance: Random guessing (~{100/num_classes:.1f}%)
""".format(k=k, num_classes=num_classes)

    elif mean_div < 0.3:
        verdict = "⚠️  MARGINAL"
        explanation = """
Subgraph distributions have SMALL differences.
SS-GNN theory applies, but learning will be DIFFICULT.

Recommendations:
  1. Try larger k (k={k+2}, k={k+4}) for better discrimination
  2. Increase hidden_dim (≥ 128) for more capacity
  3. Increase num_samples m (≥ 300) for better distribution estimation
  4. Train for many epochs (500-1000) with patience
  5. Use stronger aggregator (attention with low temperature)

Expected performance: 20-40% above random
""".format(k=k, num_classes=num_classes)

    elif mean_div < 0.5:
        verdict = "✓ MODERATE"
        explanation = """
Subgraph distributions are MODERATELY different.
SS-GNN theory applies reasonably well.

Recommendations:
  1. Standard hyperparameters should work
  2. hidden_dim ≥ 64, num_samples m ≥ 200
  3. May benefit from trying k+2 for even better discrimination
  4. Focus on architecture tuning (aggregator, encoder)

Expected performance: 40-70% above random
"""

    else:  # >= 0.5
        verdict = "✅ EXCELLENT"
        explanation = """
Subgraph distributions are HIGHLY discriminative!
SS-GNN theory applies very well to this dataset.

Recommendations:
  1. Current k={k} is good
  2. Standard hyperparameters sufficient
  3. Focus on optimization and architecture details
  4. Should achieve near state-of-the-art performance

Expected performance: 70%+ above random
""".format(k=k)

    print(f"\nVerdict: {verdict}")
    print(f"Mean JS Divergence: {mean_div:.6f}")
    print(explanation)

    # Try different k values suggestion
    if mean_div < 0.3:
        print("=" * 80)
        print("SUGGESTED NEXT STEPS")
        print("=" * 80)
        print(f"\nTry measuring divergence with larger k:")
        for test_k in [k+2, k+4, k+6]:
            if test_k <= 12:  # Don't suggest too large
                print(f"  python measure_subgraph_divergence.py --dataset {dataset_name} --k {test_k} --num_samples {num_samples}")

    return {
        'mean': mean_div,
        'std': std_div,
        'min': min_div,
        'max': max_div,
        'verdict': verdict,
        'divergences': divergences,
        'divergence_matrix': divergence_matrix
    }


def main():
    parser = argparse.ArgumentParser(
        description="Measure subgraph distribution divergence for SS-GNN applicability"
    )
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (CSL, MUTAG, PROTEINS, etc.)')
    parser.add_argument('--k', type=int, default=4,
                       help='Subgraph size (default: 4)')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of subgraphs to sample per graph (default: 100)')
    parser.add_argument('--method', type=str, default='random_walk',
                       choices=['random_walk', 'uniform'],
                       help='Sampling method (default: random_walk)')

    args = parser.parse_args()

    results = measure_divergence(
        args.dataset,
        args.k,
        args.num_samples,
        args.method
    )

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
