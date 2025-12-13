#!/usr/bin/env python3
"""
Wasserstein Distance Analysis Tool for SS-GNN

Purpose: Predict whether SS-GNN will work on a dataset by analyzing
         Wasserstein distances between subgraph distributions of different classes.

Theory: If W-distance between different classes >> W-distance within same class,
        then SS-GNN should be able to distinguish them.

Usage:
    python tools/wasserstein_analysis.py --dataset CSL --k 6 --m 100 --samples_per_class 5
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
from pathlib import Path
import sys
import json
from typing import Dict, List, Tuple
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_add_pool
import torch_geometric.nn as pyg
from torch_geometric.utils import scatter

# Import samplers
try:
    import uniform_sampler
    UNIFORM_AVAILABLE = True
except ImportError:
    UNIFORM_AVAILABLE = False
    print("Warning: uniform_sampler not available, using alternative")

try:
    import ugs_sampler
    UGS_AVAILABLE = True
except ImportError:
    UGS_AVAILABLE = False

try:
    import rwr_sampler
    RWR_AVAILABLE = True
except ImportError:
    RWR_AVAILABLE = False


class SimpleSubgraphEncoder(nn.Module):
    """Simple GIN-based encoder for subgraph embeddings"""

    def __init__(self, in_dim, hidden_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.proj = nn.Linear(in_dim, hidden_dim, bias=False)
        self.convs = nn.ModuleList()

        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
            self.convs.append(GINConv(mlp))

        self.pool = global_add_pool

    def forward(self, x, edge_index, batch):
        h = self.proj(x)
        for conv in self.convs:
            h = conv(h, edge_index)
        h_out = self.pool(h, batch)
        return h_out


def wasserstein_distance(embd1, embd2, epsilon=0.01, num_iters=100):
    """
    Approximate Wasserstein distance using Sinkhorn algorithm.
    Fully differentiable and works on GPU.

    Args:
        embd1, embd2: [m, d] tensors of embeddings
        epsilon: regularization parameter (smaller = more accurate but slower)
        num_iters: number of Sinkhorn iterations

    Returns:
        Wasserstein distance (scalar tensor)
    """
    m1, m2 = embd1.shape[0], embd2.shape[0]
    device = embd1.device

    # Uniform distributions
    a = torch.ones(m1, device=device) / m1
    b = torch.ones(m2, device=device) / m2

    # Pairwise squared distances
    C = torch.cdist(embd1, embd2, p=2) ** 2  # [m1, m2]

    # Sinkhorn iterations
    K = torch.exp(-C / epsilon)
    u = torch.ones_like(a)
    v = torch.ones_like(b)

    for _ in range(num_iters):
        u = a / (K @ v + 1e-8)
        v = b / (K.t() @ u + 1e-8)

    # Compute optimal transport plan
    pi = u.unsqueeze(1) * K * v.unsqueeze(0)

    # Wasserstein distance
    W = (pi * C).sum()

    return W.item()


def sample_subgraphs(batch, k, m, sampler='uniform', device='cpu'):
    """
    Sample subgraphs from a batch of graphs.

    Returns:
        x_global: [num_samples * k, feature_dim] - node features
        edge_index_global: [2, num_edges] - edge connectivity
        sample_id: [num_samples * k] - sample assignment for each node
    """
    num_graphs = batch.batch.max().item() + 1
    num_subgraphs = num_graphs * m

    # Sample subgraphs
    if sampler == 'uniform' and UNIFORM_AVAILABLE:
        sampler_fn = uniform_sampler.sample_batch
    elif sampler == 'ugs' and UGS_AVAILABLE:
        sampler_fn = ugs_sampler.sample_batch
    elif sampler == 'rwr' and RWR_AVAILABLE:
        sampler_fn = rwr_sampler.sample_batch
    else:
        raise ValueError(f"Sampler {sampler} not available")

    nodes_sampled, edge_index_sampled, edge_ptr, sample_ptr, edge_src_global = \
        sampler_fn(batch.edge_index.cpu(), batch.ptr.cpu(), m_per_graph=m, k=k)

    # Build global indices
    x_global = batch.x[nodes_sampled.flatten().to(device)]
    edge_index_global = torch.repeat_interleave(
        torch.arange(0, num_subgraphs),
        edge_ptr[1:] - edge_ptr[:-1]
    ) * k + edge_index_sampled
    sample_id = torch.repeat_interleave(torch.arange(0, num_subgraphs), k)

    return x_global, edge_index_global.to(device), sample_id.to(device)


def compute_subgraph_embeddings(dataset, encoder, k, m, sampler, samples_per_class, device):
    """
    Compute subgraph embeddings for each class in the dataset.

    Returns:
        embeddings_by_class: Dict[class_label, List[Tensor]]
            Each tensor is [m, hidden_dim] for one graph
    """
    # Group graphs by class
    graphs_by_class = defaultdict(list)
    for graph in dataset:
        label = graph.y.item()
        graphs_by_class[label].append(graph)

    print(f"\nDataset statistics:")
    print(f"  Total graphs: {len(dataset)}")
    print(f"  Number of classes: {len(graphs_by_class)}")
    for label, graphs in sorted(graphs_by_class.items()):
        print(f"    Class {label}: {len(graphs)} graphs")

    # Compute embeddings
    embeddings_by_class = {}
    encoder.eval()

    with torch.no_grad():
        for label, graphs in sorted(graphs_by_class.items()):
            # Sample a few graphs per class
            selected = graphs[:samples_per_class]
            embeddings_list = []

            print(f"\nProcessing class {label} ({len(selected)} graphs)...")

            for i, graph in enumerate(selected):
                # Create batch with single graph
                batch = graph.to(device)
                batch.batch = torch.zeros(batch.num_nodes, dtype=torch.long, device=device)
                batch.ptr = torch.tensor([0, batch.num_nodes], dtype=torch.long)

                # Sample subgraphs
                x_global, edge_index_global, sample_id = sample_subgraphs(
                    batch, k, m, sampler, device
                )

                # Encode subgraphs
                subgraph_emb = encoder(x_global, edge_index_global, sample_id)
                embeddings_list.append(subgraph_emb.cpu())

                if (i + 1) % 5 == 0 or i == len(selected) - 1:
                    print(f"  Processed {i+1}/{len(selected)} graphs", end='\r')

            print()  # New line
            embeddings_by_class[label] = embeddings_list

    return embeddings_by_class


def compute_wasserstein_matrix(embeddings_by_class, epsilon=0.01):
    """
    Compute pairwise Wasserstein distances between all classes.

    Returns:
        W_matrix: [num_classes, num_classes] - Wasserstein distance matrix
        intra_class_distances: Dict[label, List[float]] - within-class distances
        inter_class_distances: Dict[(label1, label2), List[float]] - between-class distances
    """
    classes = sorted(embeddings_by_class.keys())
    num_classes = len(classes)

    # Average W-distance matrix
    W_matrix = np.zeros((num_classes, num_classes))

    # Detailed distances for statistics
    intra_class_distances = defaultdict(list)
    inter_class_distances = defaultdict(list)

    print("\nComputing Wasserstein distances...")

    for i, label1 in enumerate(classes):
        for j, label2 in enumerate(classes):
            embs1 = embeddings_by_class[label1]
            embs2 = embeddings_by_class[label2]

            # Compute W-distance for all pairs
            distances = []
            for emb1 in embs1:
                for emb2 in embs2:
                    if label1 == label2 and torch.equal(emb1, emb2):
                        # Same graph - skip
                        continue
                    W = wasserstein_distance(emb1, emb2, epsilon=epsilon)
                    distances.append(W)

            avg_W = np.mean(distances) if distances else 0.0
            W_matrix[i, j] = avg_W

            # Store for statistics
            if label1 == label2:
                intra_class_distances[label1].extend(distances)
            else:
                inter_class_distances[(label1, label2)].extend(distances)

    return W_matrix, intra_class_distances, inter_class_distances


def analyze_distinguishability(W_matrix, intra_distances, inter_distances, classes):
    """
    Analyze whether classes are distinguishable based on W-distances.

    Returns:
        analysis: Dict with statistics and recommendations
    """
    # Compute statistics
    intra_values = [d for distances in intra_distances.values() for d in distances]
    inter_values = [d for distances in inter_distances.values() for d in distances]

    intra_mean = np.mean(intra_values) if intra_values else 0.0
    intra_std = np.std(intra_values) if intra_values else 0.0
    inter_mean = np.mean(inter_values) if inter_values else 0.0
    inter_std = np.std(inter_values) if inter_values else 0.0

    # Separation ratio
    if intra_mean > 0:
        separation_ratio = inter_mean / intra_mean
    else:
        separation_ratio = float('inf') if inter_mean > 0 else 1.0

    # Per-class distinguishability
    per_class_stats = {}
    for i, label1 in enumerate(classes):
        # Average distance to other classes
        inter_to_others = []
        for j, label2 in enumerate(classes):
            if label1 != label2:
                inter_to_others.append(W_matrix[i, j])

        intra = np.mean(intra_distances.get(label1, [0.0]))
        inter = np.mean(inter_to_others) if inter_to_others else 0.0

        per_class_stats[label1] = {
            'intra_mean': intra,
            'inter_mean': inter,
            'ratio': inter / intra if intra > 0 else float('inf')
        }

    # Decision criteria
    # Based on CSL results: intra ~0.0001, inter ~0.002-0.003 → ratio ~20-30x
    # SS-GNN achieved 40% (4x random baseline)

    if separation_ratio > 20:
        recommendation = "EXCELLENT - SS-GNN should work very well"
        confidence = "High"
    elif separation_ratio > 10:
        recommendation = "GOOD - SS-GNN should work"
        confidence = "Medium-High"
    elif separation_ratio > 5:
        recommendation = "MODERATE - SS-GNN may help"
        confidence = "Medium"
    elif separation_ratio > 2:
        recommendation = "WEAK - SS-GNN may provide small improvement"
        confidence = "Low"
    else:
        recommendation = "POOR - SS-GNN unlikely to help"
        confidence = "Very Low"

    analysis = {
        'intra_class': {
            'mean': intra_mean,
            'std': intra_std,
            'values': intra_values[:100]  # Sample for brevity
        },
        'inter_class': {
            'mean': inter_mean,
            'std': inter_std,
            'values': inter_values[:100]
        },
        'separation_ratio': separation_ratio,
        'recommendation': recommendation,
        'confidence': confidence,
        'per_class': per_class_stats
    }

    return analysis


def print_analysis(analysis, W_matrix, classes):
    """Pretty print the analysis results"""

    print("\n" + "="*70)
    print("WASSERSTEIN DISTANCE ANALYSIS RESULTS")
    print("="*70)

    print("\n📊 OVERALL STATISTICS:")
    print(f"  Intra-class W-distance: {analysis['intra_class']['mean']:.6f} ± {analysis['intra_class']['std']:.6f}")
    print(f"  Inter-class W-distance: {analysis['inter_class']['mean']:.6f} ± {analysis['inter_class']['std']:.6f}")
    print(f"  Separation ratio: {analysis['separation_ratio']:.2f}x")

    print("\n🎯 RECOMMENDATION:")
    print(f"  {analysis['recommendation']}")
    print(f"  Confidence: {analysis['confidence']}")

    print("\n📈 INTERPRETATION:")
    if analysis['separation_ratio'] > 20:
        print("  ✅ Classes have very different subgraph distributions")
        print("  ✅ SS-GNN should significantly outperform vanilla GNN")
        print("  ✅ Expected improvement: 3-4x over baseline")
    elif analysis['separation_ratio'] > 10:
        print("  ✅ Classes have different subgraph distributions")
        print("  ✅ SS-GNN should outperform vanilla GNN")
        print("  ✅ Expected improvement: 2-3x over baseline")
    elif analysis['separation_ratio'] > 5:
        print("  ⚠️ Classes have moderately different distributions")
        print("  ⚠️ SS-GNN may provide improvement")
        print("  ⚠️ Expected improvement: 1.5-2x over baseline")
    elif analysis['separation_ratio'] > 2:
        print("  ⚠️ Classes have weakly different distributions")
        print("  ⚠️ SS-GNN may provide small improvement")
        print("  ⚠️ Expected improvement: <1.5x over baseline")
    else:
        print("  ❌ Classes have very similar distributions")
        print("  ❌ SS-GNN unlikely to help")
        print("  ❌ Consider using vanilla GNN or other methods")

    print("\n📋 PER-CLASS STATISTICS:")
    print(f"  {'Class':<8} {'Intra-W':<12} {'Inter-W':<12} {'Ratio':<10} Status")
    print("  " + "-"*60)
    for label in sorted(classes):
        stats = analysis['per_class'][label]
        status = "✅" if stats['ratio'] > 10 else "⚠️" if stats['ratio'] > 5 else "❌"
        print(f"  {label:<8} {stats['intra_mean']:<12.6f} {stats['inter_mean']:<12.6f} "
              f"{stats['ratio']:<10.2f} {status}")

    print("\n🗺️ WASSERSTEIN DISTANCE MATRIX:")
    print("     ", end="")
    for label in classes:
        print(f"{label:>8}", end="")
    print()
    for i, label1 in enumerate(classes):
        print(f"  {label1:<3}", end="")
        for j, label2 in enumerate(classes):
            print(f"{W_matrix[i, j]:>8.5f}", end="")
        print()

    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description='Wasserstein Distance Analysis for SS-GNN')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (CSL, MUTAG, etc.)')
    parser.add_argument('--k', type=int, default=6,
                       help='Subgraph size')
    parser.add_argument('--m', type=int, default=100,
                       help='Number of samples per graph')
    parser.add_argument('--samples_per_class', type=int, default=5,
                       help='Number of graphs to sample per class')
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='Hidden dimension for encoder')
    parser.add_argument('--num_layers', type=int, default=5,
                       help='Number of GNN layers')
    parser.add_argument('--sampler', type=str, default='uniform',
                       choices=['uniform', 'ugs', 'rwr'],
                       help='Subgraph sampler to use')
    parser.add_argument('--epsilon', type=float, default=0.01,
                       help='Sinkhorn epsilon parameter')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file for results')

    args = parser.parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    print(f"\nLoading dataset: {args.dataset}")
    from gps.datasets import build_synthetic
    from gps import ExperimentConfig, ModelConfig, TrainConfig

    # Create dummy config for dataset loading
    cfg = ExperimentConfig()
    cfg.dataset_name = args.dataset
    cfg.model_config = ModelConfig()
    cfg.model_config.node_feature_dim = 5
    cfg.model_config.kwargs = {'node_feature_type': 'all_one'}
    cfg.train = TrainConfig()
    cfg.cache_dir = 'cache'
    cfg.seed = 42

    # Load dataset
    train_loader, val_loader, test_loader = build_synthetic(cfg)
    dataset = train_loader.dataset

    # Get feature dimension
    sample = dataset[0]
    feature_dim = sample.x.shape[1]

    print(f"  Feature dimension: {feature_dim}")
    print(f"  Subgraph size (k): {args.k}")
    print(f"  Samples per graph (m): {args.m}")
    print(f"  Sampler: {args.sampler}")

    # Initialize encoder
    print(f"\nInitializing encoder...")
    encoder = SimpleSubgraphEncoder(feature_dim, args.hidden_dim, args.num_layers).to(device)
    print(f"  Parameters: {sum(p.numel() for p in encoder.parameters()):,}")

    # Compute subgraph embeddings
    embeddings_by_class = compute_subgraph_embeddings(
        dataset, encoder, args.k, args.m, args.sampler, args.samples_per_class, device
    )

    # Compute Wasserstein distances
    classes = sorted(embeddings_by_class.keys())
    W_matrix, intra_distances, inter_distances = compute_wasserstein_matrix(
        embeddings_by_class, epsilon=args.epsilon
    )

    # Analyze distinguishability
    analysis = analyze_distinguishability(W_matrix, intra_distances, inter_distances, classes)

    # Print results
    print_analysis(analysis, W_matrix, classes)

    # Save to JSON if requested
    if args.output:
        output_data = {
            'dataset': args.dataset,
            'parameters': {
                'k': args.k,
                'm': args.m,
                'samples_per_class': args.samples_per_class,
                'hidden_dim': args.hidden_dim,
                'num_layers': args.num_layers,
                'sampler': args.sampler,
                'epsilon': args.epsilon
            },
            'analysis': {
                'intra_class_mean': analysis['intra_class']['mean'],
                'inter_class_mean': analysis['inter_class']['mean'],
                'separation_ratio': analysis['separation_ratio'],
                'recommendation': analysis['recommendation'],
                'confidence': analysis['confidence']
            },
            'wasserstein_matrix': W_matrix.tolist(),
            'per_class_stats': analysis['per_class']
        }

        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n✅ Results saved to: {args.output}")


if __name__ == '__main__':
    main()
