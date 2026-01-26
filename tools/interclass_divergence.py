#!/usr/bin/env python3
"""
Inter-Class Divergence Analysis Tool for SD-GNN

Computes the inter-class divergence δ_inter as defined in the paper (Definition B.7):

    μ̄^(k)_i = E_{G∈C_i}[μ^(k)_G]      (expected k-graphlet distribution for class i)
    δ_inter = min_{i≠j} ||μ̄^(k)_i - μ̄^(k)_j||_2

This metric predicts whether SD-GNN will work on a dataset:
- Large δ_inter → SD-GNN should perform well
- Small δ_inter → SD-GNN may struggle

Usage:
    python tools/interclass_divergence.py --dataset MUTAG --k 6 --m 500
    python tools/interclass_divergence.py --dataset CSL --k 6 --m 100
    python tools/interclass_divergence.py --datasets MUTAG,PTC_MR,CSL --k 6 --m 500
"""

import argparse
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'gps'))

from gps.utils.wl_vocab import compute_wl_hash, extract_subgraph_from_batch
from gps import ExperimentConfig, ModelConfig, TrainConfig, SubgraphParam
from gps.registry import get_dataset
import gps.datasets  # trigger registration


@dataclass
class DivergenceResult:
    """Results for a single dataset analysis."""
    dataset: str
    k: int
    m: int
    num_graphs: int
    num_classes: int
    num_wl_classes: int  # |G̃_k| - number of WL equivalence classes

    # Inter-class divergence (Definition B.7)
    delta_inter: float  # min_{i≠j} ||μ̄^(k)_i - μ̄^(k)_j||_2
    delta_inter_mean: float  # mean over all class pairs
    delta_inter_max: float  # max over all class pairs

    # Intra-class variance (for context)
    sigma_intra_mean: float  # mean within-class std
    sigma_intra_max: float

    # Fisher-like ratio
    fisher_ratio: float  # δ_inter / σ_intra

    # Per-class statistics
    class_sizes: Dict[int, int]
    pairwise_divergences: Dict[str, float]  # "(i,j)" -> divergence

    # Prediction
    prediction: str  # "GOOD", "MODERATE", "WEAK"
    confidence: str


class InterClassDivergenceAnalyzer:
    """
    Computes inter-class divergence on k-graphlet WL-hash distributions.

    For each graph G:
        1. Sample m connected induced k-subgraphs
        2. Compute WL-hash for each subgraph
        3. Build empirical distribution μ̂^(k)_G over WL-equivalence classes

    For each class C_i:
        4. Compute class-average distribution: μ̄^(k)_i = mean({μ̂^(k)_G : G ∈ C_i})

    Inter-class divergence:
        5. δ_inter = min_{i≠j} ||μ̄^(k)_i - μ̄^(k)_j||_2
    """

    def __init__(self, dataset_name: str, k: int = 6, m: int = 500,
                 sampler_name: str = 'ugs', seed: int = 42):
        self.dataset_name = dataset_name
        self.k = k
        self.m = m
        self.sampler_name = sampler_name
        self.seed = seed

        # Load dataset and sampler
        self._load_dataset()
        self._load_sampler()

    def _load_dataset(self):
        """Load dataset using GPS infrastructure."""
        # Handle special datasets
        model_kwargs = {}
        synthetic_datasets = [
            'CSL', 'Triangle-Parity', 'Clique-Detection', 'Multi-Clique-Detection',
            'Clique-Detection-Controlled', 'Sparse-Clique-Detection', 'K4'
        ]
        if self.dataset_name in synthetic_datasets:
            model_kwargs['node_feature_type'] = 'all_one'
            model_kwargs['max_degree'] = 64

        cfg = ExperimentConfig(
            name=f"{self.dataset_name}_divergence",
            task="Binary-Classification",
            dataset_name=self.dataset_name,
            model_name="VANILLA",
            model_config=ModelConfig(
                mpnn_type="gin",
                mpnn_layers=3,
                node_feature_dim=7,
                hidden_dim=64,
                out_dim=2,
                dropout=0.0,
                pooling="mean",
                temperature=1.0,
                subgraph_sampling=False,
                subgraph_param=SubgraphParam(k=self.k, m=self.m, pooling="mean"),
                kwargs=model_kwargs
            ),
            train=TrainConfig(
                epochs=1, train_batch_size=32, val_batch_size=32,
                lr=0.001, optimizer="adam", loss_fn="CrossEntropyLoss",
                metric="ACC", train_ratio=0.8, val_ratio=0.1
            ),
            device="cpu",
            seed=self.seed,
            num_workers=0,
            sampler=self.sampler_name,
            cache_dir="./cache",
            output_dir="./experiments"
        )

        dataset_fn = get_dataset(self.dataset_name)
        train_loader, val_loader, test_loader = dataset_fn(cfg)

        # Combine all data
        self.all_loaders = [train_loader, val_loader, test_loader]

        # Count graphs and classes
        labels = []
        for loader in self.all_loaders:
            for batch in loader:
                for i in range(batch.num_graphs):
                    if batch.y.dim() == 0:
                        labels.append(batch.y.item())
                    elif batch.y.dim() == 1:
                        labels.append(batch.y[i].item())
                    else:
                        labels.append(batch.y[i, 0].item())

        self.num_graphs = len(labels)
        self.unique_classes = sorted(set(labels))
        self.num_classes = len(self.unique_classes)

        # Handle regression (many unique values)
        if self.num_classes > self.num_graphs * 0.5:
            print(f"  Regression detected ({self.num_classes} unique values)")
            print(f"  Binning into 4 quantile-based classes...")
            labels_arr = np.array(labels)
            bins = np.quantile(labels_arr, [0, 0.25, 0.5, 0.75, 1.0])
            bins = np.unique(bins)
            self.label_bins = bins
            self.is_regression = True
            self.num_classes = len(bins) - 1
            self.unique_classes = list(range(self.num_classes))
        else:
            self.is_regression = False
            self.label_bins = None

        print(f"  Loaded {self.num_graphs} graphs, {self.num_classes} classes")

    def _load_sampler(self):
        """Load subgraph sampler."""
        if self.sampler_name == 'ugs':
            import ugs_sampler
            self.sampler = ugs_sampler
        elif self.sampler_name == 'uniform':
            import uniform_sampler
            self.sampler = uniform_sampler
        elif self.sampler_name == 'rwr':
            import rwr_sampler
            self.sampler = rwr_sampler
        else:
            raise ValueError(f"Unknown sampler: {self.sampler_name}")

    def _bin_label(self, label: float) -> int:
        """Convert continuous label to bin index."""
        if not self.is_regression:
            return int(label)
        return min(np.digitize(label, self.label_bins[1:-1]), self.num_classes - 1)

    def _compute_graph_distribution(self, batch, graph_idx: int,
                                     batch_idx: int) -> Optional[Counter]:
        """
        Compute empirical k-graphlet distribution μ̂^(k)_G for one graph.

        Returns:
            Counter mapping WL-hash -> count (not normalized yet)
        """
        try:
            # Sample subgraphs
            nodes_t, edge_index_t, edge_ptr_t, sample_ptr_t, _ = \
                self.sampler.sample_batch(
                    edge_index=batch.edge_index,
                    ptr=batch.ptr,
                    m_per_graph=self.m,
                    k=self.k,
                    mode="sample",
                    seed=self.seed + batch_idx
                )

            # Get subgraphs for this graph
            start_idx = sample_ptr_t[graph_idx].item()
            end_idx = sample_ptr_t[graph_idx + 1].item()

            counter = Counter()

            for subgraph_idx in range(start_idx, end_idx):
                subgraph_edges, num_nodes, node_features = extract_subgraph_from_batch(
                    batch, subgraph_idx, nodes_t, edge_index_t, edge_ptr_t
                )

                if num_nodes == 0:
                    continue

                wl_hash = compute_wl_hash(
                    edge_index=subgraph_edges,
                    num_nodes=num_nodes,
                    node_features=None,  # Structure only
                    num_iterations=3
                )
                counter[wl_hash] += 1

            return counter

        except Exception as e:
            return None

    def compute_distributions(self) -> Tuple[Dict[int, List[np.ndarray]], Dict[str, int]]:
        """
        Compute k-graphlet distributions for all graphs, grouped by class.

        Returns:
            distributions_by_class: {class_label: [μ̂^(k)_G1, μ̂^(k)_G2, ...]}
            wl_vocab: {wl_hash: index}
        """
        print(f"\nSampling k={self.k} graphlets (m={self.m} per graph)...")

        # First pass: collect all WL hashes to build vocabulary
        all_counters = []  # [(class_label, counter), ...]
        wl_vocab = {}

        batch_idx = 0
        for loader in self.all_loaders:
            for batch in tqdm(loader, desc="Processing batches"):
                for graph_idx in range(batch.num_graphs):
                    # Get label
                    if batch.y.dim() == 0:
                        label = batch.y.item()
                    elif batch.y.dim() == 1:
                        label = batch.y[graph_idx].item()
                    else:
                        label = batch.y[graph_idx, 0].item()

                    class_label = self._bin_label(label)

                    # Compute distribution
                    counter = self._compute_graph_distribution(batch, graph_idx, batch_idx)

                    if counter is not None and len(counter) > 0:
                        # Update vocabulary
                        for wl_hash in counter:
                            if wl_hash not in wl_vocab:
                                wl_vocab[wl_hash] = len(wl_vocab)
                        all_counters.append((class_label, counter))

                batch_idx += 1

        print(f"  Found {len(wl_vocab)} unique WL-equivalence classes")
        print(f"  Processed {len(all_counters)} graphs")

        # Second pass: convert counters to normalized distribution vectors
        num_wl_classes = len(wl_vocab)
        distributions_by_class = defaultdict(list)

        for class_label, counter in all_counters:
            # Create distribution vector
            dist = np.zeros(num_wl_classes)
            total = sum(counter.values())

            for wl_hash, count in counter.items():
                idx = wl_vocab[wl_hash]
                dist[idx] = count / total  # Normalize to probability

            distributions_by_class[class_label].append(dist)

        return dict(distributions_by_class), wl_vocab

    def compute_divergence(self) -> DivergenceResult:
        """
        Compute inter-class divergence δ_inter.

        δ_inter = min_{i≠j} ||μ̄^(k)_i - μ̄^(k)_j||_2

        where μ̄^(k)_i = mean({μ̂^(k)_G : G ∈ class i})
        """
        # Get distributions
        distributions_by_class, wl_vocab = self.compute_distributions()

        # Compute class-level average distributions
        class_means = {}  # class -> μ̄^(k)_class
        class_stds = {}   # class -> within-class std

        for cls, dists in distributions_by_class.items():
            dists_array = np.stack(dists)  # [num_graphs_in_class, num_wl_classes]
            class_means[cls] = np.mean(dists_array, axis=0)

            # Within-class variance: avg L2 distance from mean
            if len(dists) > 1:
                distances_to_mean = [np.linalg.norm(d - class_means[cls]) for d in dists]
                class_stds[cls] = np.mean(distances_to_mean)
            else:
                class_stds[cls] = 0.0

        # Compute pairwise inter-class divergences
        classes = sorted(class_means.keys())
        pairwise_divergences = {}

        for i, cls_i in enumerate(classes):
            for j, cls_j in enumerate(classes):
                if i < j:
                    divergence = np.linalg.norm(class_means[cls_i] - class_means[cls_j])
                    pairwise_divergences[f"({cls_i},{cls_j})"] = divergence

        # Compute summary statistics
        divergence_values = list(pairwise_divergences.values())
        delta_inter = min(divergence_values) if divergence_values else 0.0
        delta_inter_mean = np.mean(divergence_values) if divergence_values else 0.0
        delta_inter_max = max(divergence_values) if divergence_values else 0.0

        sigma_intra_values = list(class_stds.values())
        sigma_intra_mean = np.mean(sigma_intra_values) if sigma_intra_values else 0.0
        sigma_intra_max = max(sigma_intra_values) if sigma_intra_values else 0.0

        # Fisher ratio (using MEAN divergence, not min, for multi-class)
        # For binary classification, min = mean, so this is consistent
        fisher_ratio = delta_inter_mean / (sigma_intra_mean + 1e-10)

        # Prediction based on theory
        # Note: ratio > 2 means inter-class divergence is 2x larger than intra-class spread
        if fisher_ratio > 3.0:
            prediction = "GOOD"
            confidence = "High"
        elif fisher_ratio > 1.5:
            prediction = "MODERATE"
            confidence = "Medium"
        elif fisher_ratio > 0.8:
            prediction = "WEAK"
            confidence = "Low"
        else:
            prediction = "POOR"
            confidence = "Very Low"

        # Class sizes
        class_sizes = {cls: len(dists) for cls, dists in distributions_by_class.items()}

        return DivergenceResult(
            dataset=self.dataset_name,
            k=self.k,
            m=self.m,
            num_graphs=sum(class_sizes.values()),
            num_classes=len(classes),
            num_wl_classes=len(wl_vocab),
            delta_inter=delta_inter,
            delta_inter_mean=delta_inter_mean,
            delta_inter_max=delta_inter_max,
            sigma_intra_mean=sigma_intra_mean,
            sigma_intra_max=sigma_intra_max,
            fisher_ratio=fisher_ratio,
            class_sizes=class_sizes,
            pairwise_divergences=pairwise_divergences,
            prediction=prediction,
            confidence=confidence
        )


def print_result(result: DivergenceResult):
    """Pretty print a single result."""
    print(f"\n{'='*70}")
    print(f"INTER-CLASS DIVERGENCE ANALYSIS: {result.dataset}")
    print(f"{'='*70}")

    print(f"\n[Dataset Info]")
    print(f"  Graphs: {result.num_graphs}")
    print(f"  Classes: {result.num_classes} {dict(result.class_sizes)}")
    print(f"  WL-equivalence classes (|G̃_k|): {result.num_wl_classes}")
    print(f"  Subgraph size k: {result.k}")
    print(f"  Samples per graph m: {result.m}")

    print(f"\n[Inter-Class Divergence (Definition B.7)]")
    print(f"  δ_inter (min):  {result.delta_inter:.6f}")
    print(f"  δ_inter (mean): {result.delta_inter_mean:.6f}")
    print(f"  δ_inter (max):  {result.delta_inter_max:.6f}")

    print(f"\n[Intra-Class Variance]")
    print(f"  σ_intra (mean): {result.sigma_intra_mean:.6f}")
    print(f"  σ_intra (max):  {result.sigma_intra_max:.6f}")

    print(f"\n[Fisher-like Ratio (δ_mean / σ_intra)]")
    print(f"  {result.delta_inter_mean:.4f} / {result.sigma_intra_mean:.4f} = {result.fisher_ratio:.2f}")

    print(f"\n[Pairwise Divergences]")
    for pair, div in sorted(result.pairwise_divergences.items()):
        print(f"  {pair}: {div:.6f}")

    print(f"\n[Prediction for SD-GNN]")
    if result.prediction == "GOOD":
        print(f"  ✓ {result.prediction} (Confidence: {result.confidence})")
        print(f"    → SD-GNN should work well on this dataset")
        print(f"    → Large inter-class divergence enables learning")
    elif result.prediction == "MODERATE":
        print(f"  ⚠ {result.prediction} (Confidence: {result.confidence})")
        print(f"    → SD-GNN may work with proper hyperparameters")
        print(f"    → Consider trying different k values")
    elif result.prediction == "WEAK":
        print(f"  ⚠ {result.prediction} (Confidence: {result.confidence})")
        print(f"    → SD-GNN may struggle on this dataset")
        print(f"    → Inter-class divergence is small relative to intra-class variance")
    else:
        print(f"  ✗ {result.prediction} (Confidence: {result.confidence})")
        print(f"    → SD-GNN is unlikely to outperform vanilla GNN")
        print(f"    → Classes have very similar subgraph distributions")

    print(f"{'='*70}\n")


def generate_paper_table(results: List[DivergenceResult], output_path: str = None):
    """
    Generate a table suitable for the paper.

    Columns: Dataset | |G| | Classes | |G̃_k| | δ_inter | σ_intra | Ratio | Prediction
    """
    rows = []
    for r in results:
        rows.append({
            'Dataset': r.dataset,
            '|G|': r.num_graphs,
            'Classes': r.num_classes,
            '|G̃_k|': r.num_wl_classes,
            'δ_inter (min)': f"{r.delta_inter:.4f}",
            'δ_inter (mean)': f"{r.delta_inter_mean:.4f}",
            'σ_intra': f"{r.sigma_intra_mean:.4f}",
            'Ratio': f"{r.fisher_ratio:.2f}",
            'Prediction': r.prediction
        })

    df = pd.DataFrame(rows)

    print("\n" + "="*80)
    print("TABLE FOR PAPER: Inter-Class Divergence Analysis")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80 + "\n")

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Table saved to: {output_path}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description='Compute inter-class divergence δ_inter for SD-GNN',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single dataset:
    python tools/interclass_divergence.py --dataset MUTAG --k 6 --m 500

  Multiple datasets:
    python tools/interclass_divergence.py --datasets MUTAG,PTC_MR,CSL --k 6 --m 500

  All paper datasets:
    python tools/interclass_divergence.py --paper-datasets --k 6 --m 500
        """
    )

    parser.add_argument('--dataset', type=str, default=None,
                        help='Single dataset name')
    parser.add_argument('--datasets', type=str, default=None,
                        help='Comma-separated list of datasets')
    parser.add_argument('--paper-datasets', action='store_true',
                        help='Run on all datasets from the paper')
    parser.add_argument('--k', type=int, default=6,
                        help='Subgraph size (default: 6)')
    parser.add_argument('--m', type=int, default=500,
                        help='Samples per graph (default: 500)')
    parser.add_argument('--sampler', type=str, default='ugs',
                        choices=['ugs', 'uniform', 'rwr'],
                        help='Sampler (default: ugs)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for results')
    parser.add_argument('--output-table', type=str, default=None,
                        help='Output CSV file for paper table')

    args = parser.parse_args()

    # Determine datasets to analyze
    if args.paper_datasets:
        datasets = [
            'MUTAG', 'PTC_MR',  # Primary molecular
            'CSL',  # Synthetic
            # 'AIDS', 'PROTEINS',  # Secondary (optional)
        ]
    elif args.datasets:
        datasets = [d.strip() for d in args.datasets.split(',')]
    elif args.dataset:
        datasets = [args.dataset]
    else:
        parser.error("Must specify --dataset, --datasets, or --paper-datasets")

    # Run analysis
    results = []

    for dataset_name in datasets:
        print(f"\n{'#'*70}")
        print(f"# Analyzing: {dataset_name}")
        print(f"{'#'*70}")

        try:
            analyzer = InterClassDivergenceAnalyzer(
                dataset_name=dataset_name,
                k=args.k,
                m=args.m,
                sampler_name=args.sampler,
                seed=args.seed
            )

            result = analyzer.compute_divergence()
            results.append(result)

            print_result(result)

            # Save individual result
            if args.output:
                Path(args.output).mkdir(parents=True, exist_ok=True)
                result_path = Path(args.output) / f"{dataset_name}_k{args.k}_m{args.m}.json"
                with open(result_path, 'w') as f:
                    json.dump(asdict(result), f, indent=2)
                print(f"Result saved to: {result_path}")

        except Exception as e:
            print(f"ERROR processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Generate paper table
    if len(results) > 0:
        table_path = args.output_table
        if table_path is None and args.output:
            table_path = str(Path(args.output) / f"divergence_table_k{args.k}.csv")
        generate_paper_table(results, table_path)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for r in results:
        status = "✓" if r.prediction == "GOOD" else "⚠" if r.prediction in ["MODERATE", "WEAK"] else "✗"
        print(f"  {status} {r.dataset}: δ_inter={r.delta_inter:.4f}, ratio={r.fisher_ratio:.2f} → {r.prediction}")
    print("="*70)


if __name__ == '__main__':
    main()
