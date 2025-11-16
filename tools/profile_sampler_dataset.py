#!/usr/bin/env python3
"""
Profile subgraph samplers across entire datasets.

This tool evaluates sampler coverage across all graphs in a dataset to help determine:
1. Which sampler provides best coverage
2. What k value is appropriate for the dataset
3. How many samples_per_graph (m) are needed for good coverage

For each graph in the dataset, this tool:
1. Enumerates all distinct connected k-graphlets exactly (via BFS/DFS)
2. Samples m k-subgraphs using the specified sampler
3. Calculates coverage (% of exact graphlets found in samples)
4. Skips graphs where enumeration would be too expensive

The tool reports per-graph and aggregate statistics to help you choose:
- Best sampler: one with highest average coverage
- Best k: balance between coverage and computational cost
- Best m: minimum samples needed to achieve target coverage

Usage:
    python tools/profile_sampler_dataset.py --dataset PROTEINS --sampler rwr --k 5 --m 100
    python tools/profile_sampler_dataset.py --dataset MUTAG --sampler uniform --k 4 5 --m 50 100 200
    python tools/profile_sampler_dataset.py --dataset PROTEINS --sampler rwr ugs --k 5 6 --m 100
"""

import time
import random
import numpy as np
import torch
import argparse
import csv
from collections import Counter, defaultdict
from itertools import combinations, product
from typing import Set, Tuple, List, Dict
from multiprocessing import Pool, cpu_count
from pathlib import Path

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

try:
    import epsilon_uniform_sampler
    AVAILABLE_SAMPLERS['epsilon_uniform'] = epsilon_uniform_sampler
except ImportError:
    pass

if not AVAILABLE_SAMPLERS:
    raise ImportError("No samplers found. Please build at least one sampler extension.")

# Import PyG datasets
try:
    from torch_geometric.datasets import TUDataset, QM9, LRGBDataset, ZINC
    from torch_geometric.data import Data
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("Error: PyTorch Geometric not available. Install to run this tool.")
    exit(1)

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
                                 parallel: bool = True, max_graphlets: int = None) -> Tuple[List[Tuple[int, ...]], bool]:
    """
    Enumerate all distinct connected k-subgraphs (k-graphlets) exactly using BFS/DFS.

    Args:
        edge_index: [2, E] edge tensor
        num_nodes: Number of nodes in the graph
        k: Size of subgraphs to enumerate
        parallel: Use multiprocessing for speedup
        max_graphlets: Stop enumeration if this many graphlets are found (None = no limit)

    Returns:
        Tuple of (graphlets_list, completed)
        - graphlets_list: List of k-graphlets (sorted tuples of node IDs)
        - completed: True if enumeration completed, False if stopped early due to max_graphlets
    """
    if k > num_nodes:
        return [], True

    if k == 1:
        return [tuple([i]) for i in range(num_nodes)], True

    # Build adjacency list
    adj = defaultdict(set)
    for i in range(edge_index.size(1)):
        u, v = edge_index[0, i].item(), edge_index[1, i].item()
        adj[u].add(v)
        adj[v].add(u)

    # Convert to regular dict for pickling (multiprocessing)
    adj_dict = {k: v for k, v in adj.items()}

    if not parallel or num_nodes < 20:
        # Sequential enumeration with early stopping
        all_graphlets = set()
        for start_node in range(num_nodes):
            graphlets_local = _enumerate_from_node((start_node, k, adj_dict, num_nodes))
            all_graphlets.update(graphlets_local)

            # Check if we've exceeded the limit
            if max_graphlets is not None and len(all_graphlets) > max_graphlets:
                return sorted(list(all_graphlets))[:max_graphlets], False

        return sorted(list(all_graphlets)), True

    # Parallel enumeration
    args_list = [(node, k, adj_dict, num_nodes) for node in range(num_nodes)]

    with Pool(processes=cpu_count()) as pool:
        results = pool.map(_enumerate_from_node, args_list)

    # Merge results with early stopping check
    all_graphlets = set()
    for graphlet_set in results:
        all_graphlets.update(graphlet_set)

        # Check if we've exceeded the limit
        if max_graphlets is not None and len(all_graphlets) > max_graphlets:
            return sorted(list(all_graphlets))[:max_graphlets], False

    return sorted(list(all_graphlets)), True


def sample_subgraphs(edge_index: torch.Tensor, num_nodes: int,
                     m_samples: int, k: int,
                     p_restart: float = 0.2, seed: int = 42, epsilon: float = 0.1) -> List[Tuple[int, ...]]:
    """
    Sample k-subgraphs using the specified sampler.

    Args:
        edge_index: [2, E] edge tensor
        num_nodes: Number of nodes
        m_samples: Number of samples to draw
        k: Subgraph size
        p_restart: Restart probability (for RWR sampler only)
        seed: Random seed
        epsilon: Approximation parameter (for epsilon_uniform sampler)

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
    elif SAMPLER_NAME == 'epsilon_uniform':
        # epsilon_uniform accepts epsilon parameter
        nodes_t, edge_index_t, edge_ptr_t, sample_ptr_t, edge_src_t = \
            SAMPLER_MODULE.sample_batch(
                edge_index, ptr, m_per_graph=m_samples, k=k,
                mode="sample", seed=seed, epsilon=epsilon
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


def calculate_coverage(exact_graphlets: List[Tuple[int, ...]],
                       sampled_graphlets: List[Tuple[int, ...]]) -> Tuple[float, int, int]:
    """
    Calculate coverage: what fraction of exact graphlets were found in samples.

    Args:
        exact_graphlets: List of all exact k-graphlets
        sampled_graphlets: List of sampled k-graphlets

    Returns:
        Tuple of (coverage, num_covered, num_total)
        - coverage: Fraction of exact graphlets found in samples (0.0 to 1.0)
        - num_covered: Number of exact graphlets that appear in samples
        - num_total: Total number of exact graphlets
    """
    if len(exact_graphlets) == 0:
        return 0.0, 0, 0

    exact_set = set(exact_graphlets)
    sampled_set = set(sampled_graphlets)

    covered = exact_set & sampled_set
    coverage = len(covered) / len(exact_set)

    return coverage, len(covered), len(exact_set)


def load_dataset(dataset_name: str):
    """Load dataset by name."""
    print(f"Loading {dataset_name} dataset...")

    if dataset_name == 'QM9':
        dataset = QM9(root='data/QM9')
    elif dataset_name == 'ZINC':
        dataset = ZINC(root='data/ZINC', subset=True)
    elif dataset_name.startswith('Peptides'):
        dataset = LRGBDataset(root='data/LRGB', name=dataset_name)
    else:
        # TUDataset (PROTEINS, MUTAG, ENZYMES, DD, etc.)
        dataset = TUDataset(root='data/TUDataset', name=dataset_name)

    print(f"Loaded {len(dataset)} graphs from {dataset_name}")
    return dataset


def profile_graph(graph_idx: int, data: Data, k: int, m: int,
                  max_graphlets: int, p_restart: float, epsilon: float,
                  seed_base: int) -> Dict:
    """
    Profile a single graph: enumerate exact graphlets and calculate coverage.

    Returns:
        Dictionary with profiling results or None if graph was skipped
    """
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    num_edges = edge_index.size(1) // 2

    # Skip if k is too large
    if k > num_nodes:
        return {
            'graph_idx': graph_idx,
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'k': k,
            'm': m,
            'status': 'skipped_k_too_large',
            'exact_graphlets': 0,
            'coverage': 0.0,
            'time_exact': 0.0,
            'time_sample': 0.0
        }

    # Exact enumeration with limit
    t0 = time.perf_counter()
    exact_graphlets, completed = enumerate_k_graphlets_exact(
        edge_index, num_nodes, k, parallel=USE_PARALLEL, max_graphlets=max_graphlets
    )
    t_exact = time.perf_counter() - t0

    # Skip if too many graphlets (enumeration incomplete)
    if not completed:
        return {
            'graph_idx': graph_idx,
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'k': k,
            'm': m,
            'status': 'skipped_too_many_graphlets',
            'exact_graphlets': len(exact_graphlets),
            'coverage': 0.0,
            'time_exact': t_exact,
            'time_sample': 0.0
        }

    # Sample subgraphs
    t0 = time.perf_counter()
    seed = seed_base + graph_idx  # Different seed per graph
    sampled_graphlets = sample_subgraphs(edge_index, num_nodes, m, k,
                                         p_restart=p_restart, seed=seed, epsilon=epsilon)
    t_sample = time.perf_counter() - t0

    # Calculate coverage
    coverage, num_covered, num_total = calculate_coverage(exact_graphlets, sampled_graphlets)

    return {
        'graph_idx': graph_idx,
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'k': k,
        'm': m,
        'status': 'completed',
        'exact_graphlets': num_total,
        'covered_graphlets': num_covered,
        'valid_samples': len(sampled_graphlets),
        'coverage': coverage,
        'time_exact': t_exact,
        'time_sample': t_sample
    }


def profile_dataset(dataset_name: str, k_values: List[int], m_values: List[int],
                   sampler_name: str, max_graphs: int = None,
                   max_graphlets: int = 50000, p_restart: float = 0.2,
                   epsilon: float = 0.1, seed: int = 42,
                   output_csv: str = None):
    """
    Profile sampler coverage across entire dataset.

    Args:
        dataset_name: Name of the dataset
        k_values: List of k values to test
        m_values: List of m (samples_per_graph) values to test
        sampler_name: Name of the sampler to use
        max_graphs: Maximum number of graphs to process (None = all)
        max_graphlets: Skip graphs with more than this many exact graphlets
        p_restart: Restart probability for RWR sampler
        epsilon: Epsilon parameter for epsilon_uniform sampler
        seed: Random seed base
        output_csv: Path to save CSV results (None = don't save)
    """
    # Load dataset
    dataset = load_dataset(dataset_name)

    # Limit number of graphs if specified
    num_graphs = min(len(dataset), max_graphs) if max_graphs else len(dataset)

    print("\n" + "="*80)
    print(f"PROFILING {sampler_name.upper()} SAMPLER ON {dataset_name}")
    print("="*80)
    print(f"Number of graphs: {num_graphs} (out of {len(dataset)})")
    print(f"k values: {k_values}")
    print(f"m values: {m_values}")
    print(f"Max graphlets per graph: {max_graphlets}")
    if sampler_name == 'rwr':
        print(f"p_restart: {p_restart}")
    elif sampler_name == 'epsilon_uniform':
        print(f"epsilon: {epsilon}")
    print("="*80 + "\n")

    # Store all results
    all_results = []

    # Process each combination of k and m
    for k in k_values:
        for m in m_values:
            print(f"\n{'='*80}")
            print(f"Processing: k={k}, m={m}")
            print(f"{'='*80}")

            results_for_km = []
            skipped_k_large = 0
            skipped_too_many = 0
            completed = 0

            # Process each graph
            for graph_idx in range(num_graphs):
                if graph_idx % 50 == 0 and graph_idx > 0:
                    print(f"  Processed {graph_idx}/{num_graphs} graphs... "
                          f"(completed: {completed}, skipped: {skipped_k_large + skipped_too_many})")

                data = dataset[graph_idx]
                result = profile_graph(
                    graph_idx, data, k, m, max_graphlets,
                    p_restart, epsilon, seed
                )

                results_for_km.append(result)
                all_results.append(result)

                # Update counters
                if result['status'] == 'completed':
                    completed += 1
                elif result['status'] == 'skipped_k_too_large':
                    skipped_k_large += 1
                elif result['status'] == 'skipped_too_many_graphlets':
                    skipped_too_many += 1

            # Print summary for this k, m combination
            print(f"\nSummary for k={k}, m={m}:")
            print(f"  Completed: {completed}/{num_graphs}")
            print(f"  Skipped (k too large): {skipped_k_large}")
            print(f"  Skipped (too many graphlets): {skipped_too_many}")

            if completed > 0:
                # Calculate statistics on completed graphs
                completed_results = [r for r in results_for_km if r['status'] == 'completed']
                coverages = [r['coverage'] for r in completed_results]
                exact_counts = [r['exact_graphlets'] for r in completed_results]

                print(f"\n  Coverage statistics:")
                print(f"    Mean: {np.mean(coverages):.4f}")
                print(f"    Median: {np.median(coverages):.4f}")
                print(f"    Std: {np.std(coverages):.4f}")
                print(f"    Min: {np.min(coverages):.4f}")
                print(f"    Max: {np.max(coverages):.4f}")

                print(f"\n  Exact graphlets per graph:")
                print(f"    Mean: {np.mean(exact_counts):.1f}")
                print(f"    Median: {np.median(exact_counts):.1f}")
                print(f"    Min: {np.min(exact_counts)}")
                print(f"    Max: {np.max(exact_counts)}")

                # Show graphs with low coverage
                low_coverage = [(r['graph_idx'], r['coverage']) for r in completed_results if r['coverage'] < 0.5]
                if low_coverage:
                    print(f"\n  Graphs with coverage < 50%: {len(low_coverage)}")
                    if len(low_coverage) <= 5:
                        for gidx, cov in low_coverage:
                            print(f"    Graph {gidx}: {cov:.4f}")

    # Save to CSV if requested
    if output_csv:
        save_results_to_csv(all_results, output_csv, dataset_name, sampler_name)
        print(f"\nResults saved to: {output_csv}")

    # Print final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    # Group by k and m for final summary
    for k in k_values:
        for m in m_values:
            km_results = [r for r in all_results if r['k'] == k and r['m'] == m and r['status'] == 'completed']
            if km_results:
                coverages = [r['coverage'] for r in km_results]
                print(f"\nk={k}, m={m}: {len(km_results)} graphs")
                print(f"  Coverage: {np.mean(coverages):.4f} ± {np.std(coverages):.4f}")
                print(f"  Graphs with >90% coverage: {sum(1 for c in coverages if c > 0.9)} ({100*sum(1 for c in coverages if c > 0.9)/len(coverages):.1f}%)")
                print(f"  Graphs with >80% coverage: {sum(1 for c in coverages if c > 0.8)} ({100*sum(1 for c in coverages if c > 0.8)/len(coverages):.1f}%)")
                print(f"  Graphs with >50% coverage: {sum(1 for c in coverages if c > 0.5)} ({100*sum(1 for c in coverages if c > 0.5)/len(coverages):.1f}%)")

    print("\n" + "="*80)


def save_results_to_csv(results: List[Dict], filepath: str, dataset_name: str, sampler_name: str):
    """Save profiling results to CSV file."""
    # Ensure directory exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            'dataset', 'sampler', 'graph_idx', 'num_nodes', 'num_edges',
            'k', 'm', 'status', 'exact_graphlets', 'covered_graphlets',
            'valid_samples', 'coverage', 'time_exact', 'time_sample'
        ])

        # Data rows
        for r in results:
            writer.writerow([
                dataset_name,
                sampler_name,
                r['graph_idx'],
                r['num_nodes'],
                r['num_edges'],
                r['k'],
                r['m'],
                r['status'],
                r['exact_graphlets'],
                r.get('covered_graphlets', 0),
                r.get('valid_samples', 0),
                r['coverage'],
                r['time_exact'],
                r['time_sample']
            ])


def main():
    """Run coverage profiling on entire dataset."""
    parser = argparse.ArgumentParser(
        description='Profile subgraph sampler coverage across entire datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test single sampler with multiple k and m values
  python tools/profile_sampler_dataset.py --dataset PROTEINS --sampler rwr --k 5 6 --m 50 100 200

  # Test multiple samplers
  python tools/profile_sampler_dataset.py --dataset MUTAG --sampler rwr uniform --k 4 5 --m 100

  # Save results to CSV
  python tools/profile_sampler_dataset.py --dataset PROTEINS --sampler rwr --k 5 --m 100 --output results.csv

  # Process limited number of graphs
  python tools/profile_sampler_dataset.py --dataset QM9 --sampler rwr --k 6 --m 100 --max-graphs 200

Supported datasets: PROTEINS, MUTAG, ENZYMES, QM9, ZINC, Peptides-func, and other TUDatasets

The tool will help you determine:
  - Which sampler provides best coverage
  - What k value is appropriate for your dataset
  - How many samples per graph (m) are needed
        """)

    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (e.g., PROTEINS, QM9, MUTAG, ZINC)')
    parser.add_argument('--sampler', type=str, nargs='+', required=True,
                        help=f'Sampler(s) to test (available: {list(AVAILABLE_SAMPLERS.keys())})')
    parser.add_argument('--k', type=int, nargs='+', required=True,
                        help='List of k values to test (e.g., --k 4 5 6)')
    parser.add_argument('--m', type=int, nargs='+', required=True,
                        help='List of m (samples_per_graph) values to test (e.g., --m 50 100 200)')
    parser.add_argument('--max-graphs', type=int, default=None,
                        help='Maximum number of graphs to process (default: all)')
    parser.add_argument('--max-graphlets', type=int, default=50000,
                        help='Skip graphs with more than this many exact graphlets (default: 50000)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save CSV results (e.g., results.csv)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed base for reproducibility (default: 42)')
    parser.add_argument('--p-restart', type=float, default=0.2,
                        help='Restart probability for RWR sampler (default: 0.2)')
    parser.add_argument('--epsilon', type=float, default=0.1,
                        help='Epsilon parameter for epsilon_uniform sampler (default: 0.1)')
    parser.add_argument('--no-parallel', action='store_true',
                        help='Disable parallel enumeration')

    args = parser.parse_args()

    # Validate samplers
    for sampler in args.sampler:
        if sampler not in AVAILABLE_SAMPLERS:
            print(f"Error: Sampler '{sampler}' not available.")
            print(f"Available samplers: {list(AVAILABLE_SAMPLERS.keys())}")
            exit(1)

    # Set global flags
    global USE_PARALLEL
    USE_PARALLEL = not args.no_parallel

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Run profiling for each sampler
    for sampler in args.sampler:
        global SAMPLER_MODULE, SAMPLER_NAME
        SAMPLER_NAME = sampler
        SAMPLER_MODULE = AVAILABLE_SAMPLERS[sampler]

        # Determine output CSV path
        output_csv = None
        if args.output:
            if len(args.sampler) > 1:
                # Multiple samplers: add sampler name to filename
                base, ext = args.output.rsplit('.', 1) if '.' in args.output else (args.output, 'csv')
                output_csv = f"{base}_{sampler}.{ext}"
            else:
                output_csv = args.output

        # Run profiling
        profile_dataset(
            dataset_name=args.dataset,
            k_values=args.k,
            m_values=args.m,
            sampler_name=sampler,
            max_graphs=args.max_graphs,
            max_graphlets=args.max_graphlets,
            p_restart=args.p_restart,
            epsilon=args.epsilon,
            seed=args.seed,
            output_csv=output_csv
        )

        # Add separator between samplers
        if len(args.sampler) > 1:
            print("\n" + "="*80 + "\n")

    print("\nProfiling completed!")
    print("\nNotes:")
    print("- Coverage = fraction of exact k-graphlets found in samples")
    print("- Higher coverage indicates better sampling quality")
    print("- Good sampler should achieve high coverage with reasonable m")
    print("- Consider trade-off between coverage, k value, and computational cost")
    print("="*80)


if __name__ == "__main__":
    main()
