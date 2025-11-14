#!/usr/bin/env python3
"""
Profile subgraph samplers with coefficient of variation (CV) and exact k-graphlet enumeration.

This tool:
1. Loads graphs from PROTEINS, QM9, and Peptides-func datasets
2. Enumerates all distinct connected k-graphlets exactly (via BFS/DFS)
3. Samples many k-subgraphs using specified sampler (rwr, uniform, ugs)
4. Calculates the coefficient of variation (CV) of the empirical distribution
5. Profiles time for both exact enumeration and sampling

Two subgraphs are distinct if their node sets are different.
Subgraphs are induced connected k-subgraphs (k-graphlets).

Usage:
    python tools/profile_sampler_cv.py --sampler rwr --k 5 6 7
    python tools/profile_sampler_cv.py --sampler uniform --k 6
"""

import time
import random
import numpy as np
import torch
import argparse
from collections import Counter, defaultdict
from itertools import combinations
from typing import Set, Tuple, List, Dict
from multiprocessing import Pool, cpu_count

# Import available samplers
AVAILABLE_SAMPLERS = {}
try:
    import rwr_sampler
    AVAILABLE_SAMPLERS['rwr'] = rwr_sampler
except ImportError:
    pass

try:
    import uniform_sampler
    AVAILABLE_SAMPLERS['uniform'] = uniform_sampler
except ImportError:
    pass

try:
    import ugs_sampler
    AVAILABLE_SAMPLERS['ugs'] = ugs_sampler
except ImportError:
    pass

if not AVAILABLE_SAMPLERS:
    raise ImportError("No samplers found. Please build at least one sampler extension.")

# Import PyG datasets
try:
    from torch_geometric.datasets import TUDataset, QM9, LRGBDataset
    from torch_geometric.data import Data
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("Warning: PyTorch Geometric not available. Install to run full tests.")

# Global flags
USE_PARALLEL = True
SAMPLER_MODULE = None
SAMPLER_NAME = "rwr"


def is_connected_subgraph(nodes: Set[int], edge_index: torch.Tensor) -> bool:
    """
    Check if a set of nodes forms a connected subgraph.

    Args:
        nodes: Set of node IDs
        edge_index: [2, E] edge tensor

    Returns:
        True if the induced subgraph is connected
    """
    if len(nodes) == 0:
        return False
    if len(nodes) == 1:
        return True

    # Build adjacency list for the subgraph
    adj = defaultdict(set)
    nodes_set = set(nodes)

    for i in range(edge_index.size(1)):
        u, v = edge_index[0, i].item(), edge_index[1, i].item()
        if u in nodes_set and v in nodes_set:
            adj[u].add(v)
            adj[v].add(u)

    # BFS from first node
    start_node = next(iter(nodes))
    visited = {start_node}
    queue = [start_node]

    while queue:
        node = queue.pop(0)
        for neighbor in adj[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return len(visited) == len(nodes)


def _enumerate_from_node(args):
    """Helper for parallel BFS/DFS enumeration from a single node."""
    start_node, k, adj_dict, num_nodes = args
    graphlets_local = set()

    # DFS to find all connected k-subgraphs containing start_node
    def dfs(current_set, candidates):
        if len(current_set) == k:
            graphlets_local.add(tuple(sorted(current_set)))
            return

        # Try adding each candidate
        for node in list(candidates):
            if node <= start_node:  # Only explore if node >= start_node to avoid duplicates
                continue

            new_set = current_set | {node}
            # Find new candidates: neighbors of new_set that aren't in new_set
            new_candidates = set()
            for n in new_set:
                if n in adj_dict:
                    new_candidates.update(adj_dict[n])
            new_candidates -= new_set

            dfs(new_set, new_candidates)

    # Start DFS from start_node
    initial_candidates = set(adj_dict.get(start_node, []))
    dfs({start_node}, initial_candidates)

    return graphlets_local


def enumerate_k_graphlets_exact(edge_index: torch.Tensor, num_nodes: int, k: int,
                                 parallel: bool = True) -> List[Tuple[int, ...]]:
    """
    Enumerate all distinct connected k-subgraphs (k-graphlets) exactly using BFS/DFS.

    This uses a more efficient approach than brute force:
    - For each node, enumerate all connected k-subgraphs containing that node via DFS
    - Avoids checking disconnected combinations
    - Much faster than C(n,k) brute force in practice

    Args:
        edge_index: [2, E] edge tensor
        num_nodes: Number of nodes in the graph
        k: Size of subgraphs to enumerate
        parallel: Use multiprocessing for speedup

    Returns:
        List of k-graphlets, where each graphlet is a sorted tuple of node IDs
    """
    if k > num_nodes:
        return []

    if k == 1:
        return [tuple([i]) for i in range(num_nodes)]

    # Build adjacency list
    adj = defaultdict(set)
    for i in range(edge_index.size(1)):
        u, v = edge_index[0, i].item(), edge_index[1, i].item()
        adj[u].add(v)
        adj[v].add(u)

    # Convert to regular dict for pickling (multiprocessing)
    adj_dict = {k: v for k, v in adj.items()}

    if not parallel or num_nodes < 20:
        # Sequential enumeration
        all_graphlets = set()
        for start_node in range(num_nodes):
            graphlets_local = _enumerate_from_node((start_node, k, adj_dict, num_nodes))
            all_graphlets.update(graphlets_local)
        return sorted(list(all_graphlets))

    # Parallel enumeration
    args_list = [(node, k, adj_dict, num_nodes) for node in range(num_nodes)]

    with Pool(processes=cpu_count()) as pool:
        results = pool.map(_enumerate_from_node, args_list)

    # Merge results
    all_graphlets = set()
    for graphlet_set in results:
        all_graphlets.update(graphlet_set)

    return sorted(list(all_graphlets))


def sample_subgraphs(edge_index: torch.Tensor, num_nodes: int,
                     m_samples: int, k: int,
                     p_restart: float = 0.2, seed: int = 42) -> List[Tuple[int, ...]]:
    """
    Sample k-subgraphs using the specified sampler.

    Args:
        edge_index: [2, E] edge tensor
        num_nodes: Number of nodes
        m_samples: Number of samples to draw
        k: Subgraph size
        p_restart: Restart probability (for RWR sampler only)
        seed: Random seed

    Returns:
        List of sampled k-graphlets (sorted tuples of node IDs)
    """
    # Prepare batch input (single graph)
    ptr = torch.tensor([0, num_nodes], dtype=torch.long)

    # Sample using the specified sampler
    if SAMPLER_NAME == 'rwr':
        nodes_t, edge_index_t, edge_ptr_t, sample_ptr_t, edge_src_t = \
            SAMPLER_MODULE.sample_batch(
                edge_index, ptr, m_per_graph=m_samples, k=k,
                mode="sample", seed=seed, p_restart=p_restart
            )
    else:
        # For uniform/ugs samplers
        nodes_t, edge_index_t, edge_ptr_t, sample_ptr_t, edge_src_t = \
            SAMPLER_MODULE.sample_batch(
                edge_index, ptr, m_per_graph=m_samples, k=k,
                mode="sample", seed=seed
            )

    # Extract node sets from samples
    sampled_graphlets = []
    for i in range(m_samples):
        nodes = nodes_t[i]
        # Filter out padding (-1 values)
        valid_nodes = nodes[nodes >= 0]

        if len(valid_nodes) == k:
            graphlets_tuple = tuple(sorted(valid_nodes.tolist()))
            sampled_graphlets.append(graphlets_tuple)

    return sampled_graphlets


def calculate_cv(samples: List[Tuple[int, ...]]) -> Tuple[float, int, Dict]:
    """
    Calculate coefficient of variation of the empirical distribution.

    Args:
        samples: List of sampled graphlets (each is a tuple of node IDs)

    Returns:
        Tuple of (cv, num_unique, frequency_dict)
        - cv: Coefficient of variation (0.0 if only one unique graphlet)
        - num_unique: Number of distinct graphlets in samples
        - frequency_dict: Counter of graphlet frequencies
    """
    if len(samples) == 0:
        return float('nan'), 0, {}

    # Count frequencies
    freq_counter = Counter(samples)
    frequencies = np.array(list(freq_counter.values()))

    # Special case: only one unique graphlet means perfect uniformity (CV = 0)
    if len(frequencies) == 1:
        return 0.0, len(freq_counter), freq_counter

    # Calculate CV for multiple graphlets
    mean_freq = frequencies.mean()
    std_freq = frequencies.std(ddof=1)
    cv = std_freq / mean_freq if mean_freq > 0 else float('nan')

    return cv, len(freq_counter), freq_counter


def test_rwr_cv_small_graph():
    """Test CV calculation on a small synthetic graph."""
    print("\n" + "="*70)
    print("TEST 1: Small Synthetic Graph (Triangle)")
    print("="*70)

    # Small triangle graph
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 0],
        [1, 0, 2, 1, 0, 2]
    ], dtype=torch.long)
    num_nodes = 3
    k = 3
    m_samples = 1000

    print(f"Graph: {num_nodes} nodes (triangle)")
    print(f"k = {k}, samples = {m_samples}")

    # Exact enumeration
    print("\n1. Exact enumeration...")
    t0 = time.perf_counter()
    exact_graphlets = enumerate_k_graphlets_exact(edge_index, num_nodes, k)
    t_exact = time.perf_counter() - t0
    print(f"   Time: {t_exact:.6f} s")
    print(f"   Found {len(exact_graphlets)} distinct {k}-graphlets")
    print(f"   Graphlets: {exact_graphlets}")

    # Sampling
    print(f"\n2. {SAMPLER_NAME.upper()} sampling...")
    t0 = time.perf_counter()
    sampled_graphlets = sample_subgraphs(edge_index, num_nodes, m_samples, k)
    t_sample = time.perf_counter() - t0
    print(f"   Time: {t_sample:.6f} s")
    print(f"   Valid samples: {len(sampled_graphlets)}/{m_samples}")

    # Calculate CV
    print("\n3. Coefficient of Variation...")
    cv, num_unique, freq_dict = calculate_cv(sampled_graphlets)
    print(f"   CV = {cv:.6f}")
    if cv == 0.0 and num_unique == 1:
        print(f"   (CV=0 because only 1 unique graphlet exists - perfect uniformity)")
    print(f"   Unique graphlets in samples: {num_unique}")
    print(f"   Total distinct graphlets (exact): {len(exact_graphlets)}")

    if len(freq_dict) > 0:
        print(f"\n   Frequency distribution:")
        for graphlet, count in sorted(freq_dict.items()):
            pct = 100.0 * count / len(sampled_graphlets)
            print(f"     {graphlet}: {count} ({pct:.2f}%)")

    print(f"\n   Time ratio (sampling/exact): {t_sample/t_exact:.2f}x")


def test_rwr_cv_proteins(k_values=None, m_samples=3000):
    """Test CV calculation on PROTEINS dataset."""
    if not HAS_DATASETS:
        print("\nSkipping PROTEINS test (PyG not available)")
        return

    print("\n" + "="*70)
    print("TEST 2: PROTEINS Dataset")
    print("="*70)

    # Load PROTEINS dataset
    dataset = TUDataset(root='data/TUDataset', name='PROTEINS')
    print(f"Loaded {len(dataset)} graphs from PROTEINS")

    # Select a random medium-sized graph
    # Filter graphs by size
    valid_graphs = [(i, dataset[i]) for i in range(len(dataset))
                    if 10 <= dataset[i].num_nodes <= 40]

    if len(valid_graphs) == 0:
        print("No suitable graphs found in PROTEINS dataset")
        return

    graph_idx, data = random.choice(valid_graphs)
    edge_index = data.edge_index
    num_nodes = data.num_nodes

    print(f"\nSelected graph {graph_idx}:")
    print(f"  Nodes: {num_nodes}, Edges: {edge_index.size(1)//2}")

    # Test with different k values
    if k_values is None:
        k_values = [4, 5, 6]

    for k in k_values:
        if k > num_nodes:
            print(f"\nSkipping k={k} (larger than graph)")
            continue

        print(f"\n--- k = {k} ---")

        # Exact enumeration
        print("1. Exact enumeration...")
        t0 = time.perf_counter()
        exact_graphlets = enumerate_k_graphlets_exact(edge_index, num_nodes, k, parallel=USE_PARALLEL)
        t_exact = time.perf_counter() - t0
        print(f"   Time: {t_exact:.4f} s")
        print(f"   Found {len(exact_graphlets)} distinct {k}-graphlets")

        # Skip if too many graphlets (would take too long)
        if len(exact_graphlets) > 50000:
            print(f"   WARNING: Too many graphlets ({len(exact_graphlets)}), skipping sampling")
            continue

        # Sampling
        print(f"2. {SAMPLER_NAME.upper()} sampling...")
        t0 = time.perf_counter()
        sampled_graphlets = sample_subgraphs(edge_index, num_nodes, m_samples, k)
        t_sample = time.perf_counter() - t0
        print(f"   Time: {t_sample:.4f} s")
        print(f"   Valid samples: {len(sampled_graphlets)}/{m_samples}")

        # Calculate CV
        cv, num_unique, freq_dict = calculate_cv(sampled_graphlets)
        print(f"3. CV = {cv:.6f}")
        print(f"   Unique in samples: {num_unique}")
        print(f"   Total distinct (exact): {len(exact_graphlets)}")
        print(f"   Coverage: {100.0*num_unique/len(exact_graphlets):.2f}%")

        # Show top-5 most frequent
        if len(freq_dict) > 0:
            top_5 = freq_dict.most_common(5)
            print(f"   Top-5 most frequent:")
            for graphlet, count in top_5:
                pct = 100.0 * count / len(sampled_graphlets)
                print(f"     {graphlet}: {count} ({pct:.2f}%)")

        print(f"   Time ratio (sampling/exact): {t_sample/t_exact:.2f}x")


def test_rwr_cv_qm9(k_values=None, m_samples=3000):
    """Test CV calculation on QM9 dataset."""
    if not HAS_DATASETS:
        print("\nSkipping QM9 test (PyG not available)")
        return

    print("\n" + "="*70)
    print("TEST 3: QM9 Dataset")
    print("="*70)

    # Load QM9 dataset
    print("Loading QM9 dataset...")
    dataset = QM9(root='data/QM9')
    print(f"Loaded {len(dataset)} molecules from QM9")

    # Select a random medium-sized molecule
    valid_mols = [(i, dataset[i]) for i in range(min(1000, len(dataset)))
                  if 8 <= dataset[i].num_nodes <= 25]

    if len(valid_mols) == 0:
        print("No suitable molecules found in QM9 dataset")
        return

    mol_idx, data = random.choice(valid_mols)
    edge_index = data.edge_index
    num_nodes = data.num_nodes

    print(f"\nSelected molecule {mol_idx}:")
    print(f"  Atoms: {num_nodes}, Bonds: {edge_index.size(1)//2}")

    # Test with different k values
    if k_values is None:
        k_values = [5, 6, 7]

    for k in k_values:
        if k > num_nodes:
            print(f"\nSkipping k={k} (larger than molecule)")
            continue

        print(f"\n--- k = {k} ---")

        # Exact enumeration
        print("1. Exact enumeration...")
        t0 = time.perf_counter()
        exact_graphlets = enumerate_k_graphlets_exact(edge_index, num_nodes, k, parallel=USE_PARALLEL)
        t_exact = time.perf_counter() - t0
        print(f"   Time: {t_exact:.4f} s")
        print(f"   Found {len(exact_graphlets)} distinct {k}-graphlets")

        # Skip if too many graphlets
        if len(exact_graphlets) > 100000:
            print(f"   WARNING: Too many graphlets ({len(exact_graphlets)}), skipping sampling")
            continue

        # Sampling
        print(f"2. {SAMPLER_NAME.upper()} sampling...")
        t0 = time.perf_counter()
        sampled_graphlets = sample_subgraphs(edge_index, num_nodes, m_samples, k)
        t_sample = time.perf_counter() - t0
        print(f"   Time: {t_sample:.4f} s")
        print(f"   Valid samples: {len(sampled_graphlets)}/{m_samples}")

        # Calculate CV
        cv, num_unique, freq_dict = calculate_cv(sampled_graphlets)
        print(f"3. CV = {cv:.6f}")
        print(f"   Unique in samples: {num_unique}")
        print(f"   Total distinct (exact): {len(exact_graphlets)}")
        print(f"   Coverage: {100.0*num_unique/len(exact_graphlets):.2f}%")

        # Show distribution statistics
        if len(freq_dict) > 0:
            freqs = np.array(list(freq_dict.values()))
            print(f"   Frequency stats: min={freqs.min()}, max={freqs.max()}, "
                  f"median={np.median(freqs):.1f}")

            top_5 = freq_dict.most_common(5)
            print(f"   Top-5 most frequent:")
            for graphlet, count in top_5:
                pct = 100.0 * count / len(sampled_graphlets)
                print(f"     {graphlet}: {count} ({pct:.2f}%)")

        print(f"   Time ratio (sampling/exact): {t_sample/t_exact:.2f}x")


def test_rwr_cv_peptides(k_values=None, m_samples=3000):
    """Test CV calculation on Peptides-func dataset."""
    if not HAS_DATASETS:
        print("\nSkipping Peptides-func test (PyG not available)")
        return

    print("\n" + "="*70)
    print("TEST 4: Peptides-func Dataset (LRGB)")
    print("="*70)

    # Load Peptides-func dataset
    print("Loading Peptides-func dataset...")
    try:
        dataset = LRGBDataset(root='data/LRGB', name='Peptides-func')
    except Exception as e:
        print(f"Failed to load Peptides-func: {e}")
        return

    print(f"Loaded {len(dataset)} peptides from Peptides-func")

    # Select a random medium-sized peptide
    valid_peptides = [(i, dataset[i]) for i in range(min(500, len(dataset)))
                      if 15 <= dataset[i].num_nodes <= 50]

    if len(valid_peptides) == 0:
        print("No suitable peptides found in Peptides-func dataset")
        return

    pep_idx, data = random.choice(valid_peptides)
    edge_index = data.edge_index
    num_nodes = data.num_nodes

    print(f"\nSelected peptide {pep_idx}:")
    print(f"  Nodes: {num_nodes}, Edges: {edge_index.size(1)//2}")

    # Test with different k values
    if k_values is None:
        k_values = [5, 6, 7]

    for k in k_values:
        if k > num_nodes:
            print(f"\nSkipping k={k} (larger than peptide)")
            continue

        print(f"\n--- k = {k} ---")

        # Exact enumeration
        print("1. Exact enumeration...")
        t0 = time.perf_counter()
        exact_graphlets = enumerate_k_graphlets_exact(edge_index, num_nodes, k, parallel=USE_PARALLEL)
        t_exact = time.perf_counter() - t0
        print(f"   Time: {t_exact:.4f} s")
        print(f"   Found {len(exact_graphlets)} distinct {k}-graphlets")

        # Skip if too many graphlets
        if len(exact_graphlets) > 100000:
            print(f"   WARNING: Too many graphlets ({len(exact_graphlets)}), skipping sampling")
            continue

        # Sampling
        print(f"2. {SAMPLER_NAME.upper()} sampling...")
        t0 = time.perf_counter()
        sampled_graphlets = sample_subgraphs(edge_index, num_nodes, m_samples, k)
        t_sample = time.perf_counter() - t0
        print(f"   Time: {t_sample:.4f} s")
        print(f"   Valid samples: {len(sampled_graphlets)}/{m_samples}")

        # Calculate CV
        cv, num_unique, freq_dict = calculate_cv(sampled_graphlets)
        print(f"3. CV = {cv:.6f}")
        print(f"   Unique in samples: {num_unique}")
        print(f"   Total distinct (exact): {len(exact_graphlets)}")
        print(f"   Coverage: {100.0*num_unique/len(exact_graphlets):.2f}%")

        # Show distribution statistics
        if len(freq_dict) > 0:
            freqs = np.array(list(freq_dict.values()))
            print(f"   Frequency stats: min={freqs.min()}, max={freqs.max()}, "
                  f"median={np.median(freqs):.1f}")

            top_5 = freq_dict.most_common(5)
            print(f"   Top-5 most frequent:")
            for graphlet, count in top_5:
                pct = 100.0 * count / len(sampled_graphlets)
                print(f"     {graphlet}: {count} ({pct:.2f}%)")

        print(f"   Time ratio (sampling/exact): {t_sample/t_exact:.2f}x")


def main():
    """Run all tests."""
    parser = argparse.ArgumentParser(
        description='Profile subgraph samplers with CV and exact k-graphlet enumeration')
    parser.add_argument('--sampler', type=str, default='rwr',
                        choices=list(AVAILABLE_SAMPLERS.keys()),
                        help=f'Sampler to use (available: {list(AVAILABLE_SAMPLERS.keys())})')
    parser.add_argument('--k', type=int, nargs='+', default=None,
                        help='List of k values to test (e.g., --k 4 5 6)')
    parser.add_argument('--num-samples', type=int, default=3000,
                        help='Number of subgraphs to sample per graph (default: 3000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--no-parallel', action='store_true',
                        help='Disable parallel enumeration')
    args = parser.parse_args()

    # Set global flags
    global USE_PARALLEL, SAMPLER_MODULE, SAMPLER_NAME
    USE_PARALLEL = not args.no_parallel
    SAMPLER_NAME = args.sampler
    SAMPLER_MODULE = AVAILABLE_SAMPLERS[args.sampler]

    print("\n" + "="*70)
    print(f"Sampler Profiling Tool: {SAMPLER_NAME.upper()}")
    print("Testing CV with exact k-graphlet enumeration")
    if USE_PARALLEL:
        print(f"Parallel enumeration: ENABLED ({cpu_count()} CPUs)")
    else:
        print("Parallel enumeration: DISABLED")
    if args.k:
        print(f"k values: {args.k}")
    print(f"Samples per graph: {args.num_samples}")
    print(f"Random seed: {args.seed}")
    print("="*70)

    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Run tests
    test_rwr_cv_small_graph()
    test_rwr_cv_proteins(k_values=args.k, m_samples=args.num_samples)
    test_rwr_cv_qm9(k_values=args.k, m_samples=args.num_samples)
    test_rwr_cv_peptides(k_values=args.k, m_samples=args.num_samples)

    print("\n" + "="*70)
    print("All tests completed!")
    print("="*70)
    print("\nNotes:")
    print("- CV (Coefficient of Variation) = std / mean of frequencies")
    print("- Lower CV indicates more uniform sampling")
    print("- Exact enumeration is exponential: C(n,k) combinations to check")
    print("- Parallel enumeration speeds up by using all CPU cores")
    print("- For large graphs/k, enumeration time >> sampling time")
    print("- Consider using estimation methods for k > 8 or n > 50")
    print("="*70)


if __name__ == "__main__":
    main()
