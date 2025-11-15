#!/usr/bin/env python3
"""
Profile subgraph samplers for validity and performance across entire datasets.

This tool:
1. Loads graphs from TUDataset, QM9, or Peptides-func datasets
2. Processes each graph with the specified sampler
3. Tracks which graphs produce invalid samples (with -1 padding nodes)
4. Measures total processing time for the entire dataset
5. Reports statistics on validity and performance

Usage:
    python tools/profile_sampler_validity.py --dataset PROTEINS --sampler rwr --k 8 --m 100
    python tools/profile_sampler_validity.py --dataset QM9 --sampler uniform --k 6 --m 50
    python tools/profile_sampler_validity.py --dataset MUTAG --sampler rwr --k 5 --m 200 --seed 42
"""

import time
import argparse
import torch
from typing import Dict, List, Tuple

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
    from torch_geometric.datasets import TUDataset, QM9, LRGBDataset
    from torch_geometric.data import Data
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("Error: PyTorch Geometric not available. Install to use this tool.")
    exit(1)


def check_graph_validity(nodes_sampled: torch.Tensor, m_per_graph: int) -> Tuple[bool, int]:
    """
    Check if any samples from a graph have invalid nodes (-1).

    Args:
        nodes_sampled: [m_per_graph, k] tensor of sampled nodes
        m_per_graph: Number of samples per graph

    Returns:
        Tuple of (has_invalid, num_invalid_samples)
        - has_invalid: True if any sample contains -1 nodes
        - num_invalid_samples: Count of samples with -1 nodes
    """
    num_invalid = 0
    for sample_idx in range(nodes_sampled.size(0)):
        if (nodes_sampled[sample_idx] == -1).any():
            num_invalid += 1

    return num_invalid > 0, num_invalid


def profile_dataset(dataset_name: str, sampler_name: str, k: int, m_per_graph: int,
                   p_restart: float = 0.2, seed: int = 42, epsilon: float = 0.1) -> Dict:
    """
    Profile sampler on an entire dataset.

    Args:
        dataset_name: Name of dataset (PROTEINS, QM9, MUTAG, etc.)
        sampler_name: Name of sampler to use
        k: Subgraph size
        m_per_graph: Number of samples per graph
        p_restart: Restart probability (for RWR sampler)
        seed: Random seed
        epsilon: Approximation parameter (for epsilon_uniform sampler)

    Returns:
        Dictionary with profiling results
    """
    # Load dataset
    print(f"Loading {dataset_name} dataset...")
    if dataset_name == 'QM9':
        dataset = QM9(root='data/QM9')
    elif dataset_name.startswith('Peptides'):
        dataset = LRGBDataset(root='data/LRGB', name=dataset_name)
    else:
        # TUDataset
        dataset = TUDataset(root='data/TUDataset', name=dataset_name)

    print(f"Loaded {len(dataset)} graphs from {dataset_name}")
    print(f"Sampler: {sampler_name}")
    print(f"Parameters: k={k}, m_per_graph={m_per_graph}, seed={seed}")
    if sampler_name == 'rwr':
        print(f"            p_restart={p_restart}")
    elif sampler_name == 'epsilon_uniform':
        print(f"            epsilon={epsilon}")
    print()

    # Get sampler module
    sampler_module = AVAILABLE_SAMPLERS[sampler_name]

    # Track results
    invalid_graph_indices = []
    invalid_sample_counts = []  # Number of invalid samples per graph with issues
    graph_sizes = []  # (num_nodes, num_edges) for graphs with invalid samples
    total_graphs = len(dataset)

    # Start timing
    print("Processing graphs...")
    start_time = time.perf_counter()

    # Process each graph
    for graph_idx in range(len(dataset)):
        graph = dataset[graph_idx]

        # Create ptr for single graph
        ptr = torch.tensor([0, graph.num_nodes], dtype=torch.long)

        # Sample from this graph
        try:
            if sampler_name == 'rwr':
                nodes_t, edge_index_t, edge_ptr_t, sample_ptr_t, edge_src_t = \
                    sampler_module.sample_batch(
                        graph.edge_index, ptr, m_per_graph=m_per_graph, k=k,
                        mode="sample", seed=seed, p_restart=p_restart
                    )
            elif sampler_name == 'epsilon_uniform':
                nodes_t, edge_index_t, edge_ptr_t, sample_ptr_t, edge_src_t = \
                    sampler_module.sample_batch(
                        graph.edge_index, ptr, m_per_graph=m_per_graph, k=k,
                        mode="sample", seed=seed, epsilon=epsilon
                    )
            else:
                nodes_t, edge_index_t, edge_ptr_t, sample_ptr_t, edge_src_t = \
                    sampler_module.sample_batch(
                        graph.edge_index, ptr, m_per_graph=m_per_graph, k=k,
                        mode="sample", seed=seed
                    )
        except Exception as e:
            print(f"Graph {graph_idx}: ERROR during sampling - {e}")
            invalid_graph_indices.append(graph_idx)
            invalid_sample_counts.append(m_per_graph)  # All samples failed
            graph_sizes.append((graph.num_nodes, graph.num_edges))
            continue

        # Check for invalid samples
        has_invalid, num_invalid = check_graph_validity(nodes_t, m_per_graph)

        if has_invalid:
            invalid_graph_indices.append(graph_idx)
            invalid_sample_counts.append(num_invalid)
            graph_sizes.append((graph.num_nodes, graph.num_edges))

        # Progress indicator - adaptive frequency based on dataset size
        progress_interval = max(1, total_graphs // 10)  # Print ~10 times
        if (graph_idx + 1) % progress_interval == 0 or (graph_idx + 1) == total_graphs:
            elapsed = time.perf_counter() - start_time
            rate = (graph_idx + 1) / elapsed
            pct = 100.0 * (graph_idx + 1) / total_graphs
            print(f"  Processed {graph_idx + 1}/{total_graphs} ({pct:.1f}%) - "
                  f"{rate:.1f} graphs/sec, {len(invalid_graph_indices)} invalid so far)")

    # End timing
    total_time = time.perf_counter() - start_time

    # Compile results
    results = {
        'dataset_name': dataset_name,
        'sampler_name': sampler_name,
        'k': k,
        'm_per_graph': m_per_graph,
        'seed': seed,
        'total_graphs': total_graphs,
        'invalid_graph_indices': invalid_graph_indices,
        'invalid_sample_counts': invalid_sample_counts,
        'graph_sizes': graph_sizes,
        'total_time': total_time,
        'graphs_per_sec': total_graphs / total_time,
    }

    if sampler_name == 'rwr':
        results['p_restart'] = p_restart
    elif sampler_name == 'epsilon_uniform':
        results['epsilon'] = epsilon

    return results


def print_results(results: Dict):
    """Print profiling results in a formatted way."""
    print("\n" + "="*80)
    print("PROFILING RESULTS")
    print("="*80)

    print(f"\nDataset: {results['dataset_name']}")
    print(f"Sampler: {results['sampler_name']}")
    print(f"Parameters: k={results['k']}, m_per_graph={results['m_per_graph']}, seed={results['seed']}")
    if 'p_restart' in results:
        print(f"            p_restart={results['p_restart']}")
    if 'epsilon' in results:
        print(f"            epsilon={results['epsilon']}")

    print(f"\n{'─'*80}")
    print("PERFORMANCE")
    print(f"{'─'*80}")
    print(f"Total graphs processed: {results['total_graphs']}")
    print(f"Total dataset processing time: {results['total_time']:.2f} seconds ({results['total_time']/60:.2f} minutes)")
    print(f"Processing rate: {results['graphs_per_sec']:.1f} graphs/second")
    print(f"Average time per graph: {results['total_time']/results['total_graphs']*1000:.2f} ms")

    print(f"\n{'─'*80}")
    print("VALIDITY")
    print(f"{'─'*80}")
    num_invalid_graphs = len(results['invalid_graph_indices'])
    print(f"Graphs with invalid samples: {num_invalid_graphs}/{results['total_graphs']} "
          f"({100.0*num_invalid_graphs/results['total_graphs']:.2f}%)")

    if num_invalid_graphs > 0:
        total_invalid_samples = sum(results['invalid_sample_counts'])
        total_samples = results['total_graphs'] * results['m_per_graph']
        print(f"Total invalid samples: {total_invalid_samples}/{total_samples} "
              f"({100.0*total_invalid_samples/total_samples:.2f}%)")

        print(f"\nInvalid samples per affected graph:")
        print(f"  Min: {min(results['invalid_sample_counts'])}")
        print(f"  Max: {max(results['invalid_sample_counts'])}")
        print(f"  Avg: {sum(results['invalid_sample_counts'])/len(results['invalid_sample_counts']):.1f}")

        # Show first 20 invalid graphs
        print(f"\nFirst {min(20, num_invalid_graphs)} graphs with invalid samples:")
        print(f"  {'Graph':>6} | {'Nodes':>6} | {'Edges':>6} | {'Invalid':>8}")
        print(f"  {'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}")
        for i in range(min(20, num_invalid_graphs)):
            idx = results['invalid_graph_indices'][i]
            num_nodes, num_edges = results['graph_sizes'][i]
            num_invalid = results['invalid_sample_counts'][i]
            print(f"  {idx:>6} | {num_nodes:>6} | {num_edges:>6} | "
                  f"{num_invalid}/{results['m_per_graph']}")

        if num_invalid_graphs > 20:
            print(f"  ... and {num_invalid_graphs - 20} more")

        # Size statistics for invalid graphs
        if results['graph_sizes']:
            node_counts = [s[0] for s in results['graph_sizes']]
            edge_counts = [s[1] for s in results['graph_sizes']]
            print(f"\nSize statistics for graphs with invalid samples:")
            print(f"  Nodes: min={min(node_counts)}, max={max(node_counts)}, "
                  f"avg={sum(node_counts)/len(node_counts):.1f}")
            print(f"  Edges: min={min(edge_counts)}, max={max(edge_counts)}, "
                  f"avg={sum(edge_counts)/len(edge_counts):.1f}")
    else:
        print("\nAll samples were valid! ✓")

    print("\n" + "="*80)


def main():
    """Run the profiling tool."""
    parser = argparse.ArgumentParser(
        description='Profile sampler validity and performance on datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/profile_sampler_validity.py --dataset PROTEINS --sampler rwr --k 8 --m 100
  python tools/profile_sampler_validity.py --dataset QM9 --sampler uniform --k 6 --m 50
  python tools/profile_sampler_validity.py --dataset MUTAG --sampler rwr --k 5 --m 200

Available TUDatasets: PROTEINS, MUTAG, ENZYMES, DD, COLLAB, etc.
        """
    )
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (e.g., PROTEINS, QM9, MUTAG, Peptides-func)')
    parser.add_argument('--sampler', type=str, required=True,
                       choices=list(AVAILABLE_SAMPLERS.keys()),
                       help=f'Sampler to use (available: {list(AVAILABLE_SAMPLERS.keys())})')
    parser.add_argument('--k', type=int, required=True,
                       help='Subgraph size (number of nodes per sample)')
    parser.add_argument('--m', dest='m_per_graph', type=int, required=True,
                       help='Number of samples per graph')
    parser.add_argument('--p-restart', type=float, default=0.2,
                       help='Restart probability for RWR sampler (default: 0.2)')
    parser.add_argument('--epsilon', type=float, default=0.1,
                       help='Approximation parameter for epsilon_uniform sampler (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)

    print("\n" + "="*80)
    print("SAMPLER VALIDITY & PERFORMANCE PROFILER")
    print("="*80)

    # Run profiling
    results = profile_dataset(
        dataset_name=args.dataset,
        sampler_name=args.sampler,
        k=args.k,
        m_per_graph=args.m_per_graph,
        p_restart=args.p_restart,
        seed=args.seed,
        epsilon=args.epsilon
    )

    # Print results
    print_results(results)

    print("\nDone!")


if __name__ == "__main__":
    main()
