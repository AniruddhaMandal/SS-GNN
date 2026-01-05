"""
Graphlet-Label Correlation Analysis Tool
Level 1: Individual Graphlet Type Analysis

Analyzes which specific k-graphlet types correlate with target labels.

Usage:
    python tools/graphlet_analysis.py --dataset MUTAG --k 6 --m 1000 --sampler ugs
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter, defaultdict
from scipy.stats import mannwhitneyu, chi2_contingency
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'gps'))

# Import GPS components
from gps.utils.wl_vocab import compute_wl_hash, extract_subgraph_from_batch
from gps import ExperimentConfig, ModelConfig, TrainConfig, SubgraphParam
from gps.registry import get_dataset
import gps.datasets  # Import to trigger registration


class GraphletAnalyzer:
    """Level 1: Individual graphlet type analysis."""

    def __init__(self, dataset_name, k=6, m=1000, sampler_name='ugs',
                 use_node_features=False, seed=42):
        """
        Args:
            dataset_name: Name of dataset (e.g., 'MUTAG', 'ZINC')
            k: Subgraph size
            m: Number of subgraphs to sample per graph
            sampler_name: 'ugs', 'uniform', 'rwr'
            use_node_features: Whether to include node features in WL hash
            seed: Random seed
        """
        self.dataset_name = dataset_name
        self.k = k
        self.m = m
        self.sampler_name = sampler_name
        self.use_node_features = use_node_features
        self.seed = seed

        print(f"\n{'='*80}")
        print(f"GRAPHLET-LABEL CORRELATION ANALYSIS")
        print(f"{'='*80}")
        print(f"Dataset: {dataset_name}")
        print(f"k-graphlet size: {k}")
        print(f"Samples per graph: {m}")
        print(f"Sampler: {sampler_name}")
        print(f"Use node features: {use_node_features}")
        print(f"{'='*80}\n")

        # Load dataset and sampler
        self._load_dataset()
        self._load_sampler()

        # Build graphlet count matrix
        print("Building graphlet count matrix...")
        self.wl_vocab, self.graphlet_matrix, self.labels = self._build_graphlet_matrix()

        print(f"✓ Found {len(self.wl_vocab)} unique graphlet types")
        print(f"✓ Processed {len(self.labels)} graphs")
        print(f"✓ Labels: {np.unique(self.labels)}\n")

    def _load_dataset(self):
        """Load dataset using GPS infrastructure."""

        # Special handling for synthetic datasets
        model_kwargs = {}
        if self.dataset_name in ['CSL', 'Triangle-Parity', 'Clique-Detection']:
            model_kwargs['node_feature_type'] = 'lap_pe'  # Laplacian PE for CSL
            model_kwargs['lap_pe_dim'] = 8

        # Create minimal config
        cfg = ExperimentConfig(
            name=f"{self.dataset_name}_analysis",
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
                epochs=1,
                train_batch_size=32,
                val_batch_size=32,
                lr=0.001,
                optimizer="adam",
                loss_fn="CrossEntropyLoss",
                metric="ACC",
                train_ratio=0.8,
                val_ratio=0.1
            ),
            device="cpu",
            seed=self.seed,
            sampler=self.sampler_name,
            cache_dir="./cache",
            output_dir="./experiments"
        )

        # Load dataset
        try:
            dataset_fn = get_dataset(self.dataset_name)
            train_loader, val_loader, test_loader = dataset_fn(cfg)

            # Combine all data for analysis
            self.dataloader = train_loader

            # Get all graphs and labels
            self.all_graphs = []
            self.all_labels = []

            for loader in [train_loader, val_loader, test_loader]:
                for batch in loader:
                    for i in range(batch.num_graphs):
                        self.all_graphs.append(batch)

                        # Handle different label formats
                        if batch.y.dim() == 0:
                            # Single scalar
                            label = batch.y.item()
                        elif batch.y.dim() == 1:
                            # Single label per graph
                            label = batch.y[i].item()
                        elif batch.y.dim() == 2 and batch.y.shape[1] > 1:
                            # Multi-label: convert to single label by taking first positive or majority
                            # For now, skip multi-label datasets
                            print(f"\n⚠ Warning: {self.dataset_name} is multi-label (shape: {batch.y.shape})")
                            print("Multi-label classification not yet supported by this tool.")
                            print("Recommendation: Analyze per-label separately or use regression datasets.")
                            raise ValueError(f"Multi-label classification not supported. Dataset has {batch.y.shape[1]} labels.")
                        else:
                            label = batch.y[i].item()

                        self.all_labels.append(label)

            print(f"✓ Loaded {len(self.all_labels)} graphs from {self.dataset_name}")

        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise

    def _load_sampler(self):
        """Load sampler based on name."""
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

        print(f"✓ Loaded sampler: {self.sampler_name}")

    def _build_graphlet_matrix(self):
        """Sample graphlets and build count matrix."""
        wl_vocab = {}
        graphlet_counts_list = []
        labels = []

        for batch_idx, batch in enumerate(tqdm(self.dataloader, desc="Sampling graphlets")):
            # Sample subgraphs
            try:
                nodes_t, edge_index_t, edge_ptr_t, sample_ptr_t, edge_src_t = \
                    self.sampler.sample_batch(
                        edge_index=batch.edge_index,
                        ptr=batch.ptr,
                        m_per_graph=self.m,
                        k=self.k,
                        mode="sample",
                        seed=self.seed + batch_idx
                    )

                # Process each graph in batch
                for graph_idx in range(batch.num_graphs):
                    # Get subgraphs for this graph
                    start_idx = sample_ptr_t[graph_idx].item()
                    end_idx = sample_ptr_t[graph_idx + 1].item()

                    counter = Counter()

                    for subgraph_idx in range(start_idx, end_idx):
                        # Extract subgraph
                        subgraph_edges, num_nodes, node_features = extract_subgraph_from_batch(
                            batch, subgraph_idx, nodes_t, edge_index_t, edge_ptr_t
                        )

                        if num_nodes == 0:
                            continue

                        # Compute WL hash
                        wl_hash = compute_wl_hash(
                            edge_index=subgraph_edges,
                            num_nodes=num_nodes,
                            node_features=node_features if self.use_node_features else None,
                            num_iterations=3
                        )

                        # Add to vocabulary
                        if wl_hash not in wl_vocab:
                            wl_vocab[wl_hash] = len(wl_vocab)

                        counter[wl_hash] += 1

                    graphlet_counts_list.append(counter)
                    labels.append(batch.y[graph_idx].item() if batch.y.dim() > 0 else batch.y.item())

            except Exception as e:
                print(f"\nError sampling batch {batch_idx}: {e}")
                continue

        # Convert to matrix
        num_graphs = len(graphlet_counts_list)
        num_graphlet_types = len(wl_vocab)

        matrix = np.zeros((num_graphs, num_graphlet_types))

        for i, counter in enumerate(graphlet_counts_list):
            for wl_hash, count in counter.items():
                j = wl_vocab[wl_hash]
                matrix[i, j] = count

        return wl_vocab, matrix, np.array(labels)

    # ========== Level 1: Statistical Analysis ==========

    def compute_statistics(self):
        """Compute presence rate, avg count per graphlet per class."""
        num_classes = len(np.unique(self.labels))
        stats = []

        # Reverse vocab for lookup
        id_to_hash = {v: k for k, v in self.wl_vocab.items()}

        for graphlet_idx in range(self.graphlet_matrix.shape[1]):
            counts = self.graphlet_matrix[:, graphlet_idx]
            wl_hash = id_to_hash.get(graphlet_idx, f"unknown_{graphlet_idx}")

            row = {
                'graphlet_id': graphlet_idx,
                'wl_hash': wl_hash[:16] + "..."  # Truncate for display
            }

            for c in range(num_classes):
                class_mask = self.labels == c
                class_counts = counts[class_mask]

                row[f'presence_rate_c{c}'] = np.mean(class_counts > 0)
                row[f'avg_count_c{c}'] = np.mean(class_counts)
                row[f'std_count_c{c}'] = np.std(class_counts)
                row[f'max_count_c{c}'] = np.max(class_counts)

            stats.append(row)

        return pd.DataFrame(stats)

    def test_significance(self):
        """Statistical tests for each graphlet."""
        results = []

        # Reverse vocab for lookup
        id_to_hash = {v: k for k, v in self.wl_vocab.items()}

        # Get unique classes
        unique_classes = np.unique(self.labels)

        if len(unique_classes) != 2:
            print(f"Warning: Currently only supports binary classification. Found {len(unique_classes)} classes.")
            # For multi-class, use first two classes
            c0, c1 = unique_classes[0], unique_classes[1]
        else:
            c0, c1 = unique_classes[0], unique_classes[1]

        for graphlet_idx in range(self.graphlet_matrix.shape[1]):
            counts = self.graphlet_matrix[:, graphlet_idx]
            wl_hash = id_to_hash.get(graphlet_idx, f"unknown_{graphlet_idx}")

            # Separate by class
            class_0_counts = counts[self.labels == c0]
            class_1_counts = counts[self.labels == c1]

            # Skip if no variation
            if np.std(counts) < 1e-6:
                continue

            # Mann-Whitney U test
            try:
                stat, p_value = mannwhitneyu(class_0_counts, class_1_counts, alternative='two-sided')
            except:
                p_value = 1.0

            # Effect size (Cohen's d)
            mean_0 = np.mean(class_0_counts)
            mean_1 = np.mean(class_1_counts)
            mean_diff = mean_1 - mean_0

            pooled_std = np.sqrt((np.var(class_0_counts) + np.var(class_1_counts)) / 2)
            cohens_d = mean_diff / (pooled_std + 1e-8)

            # Presence difference
            presence_0 = np.mean(class_0_counts > 0)
            presence_1 = np.mean(class_1_counts > 0)
            presence_diff = presence_1 - presence_0

            results.append({
                'graphlet_id': graphlet_idx,
                'wl_hash': wl_hash[:16] + "...",
                'p_value': p_value,
                'cohens_d': cohens_d,
                'effect_size': 'large' if abs(cohens_d) > 0.8 else ('medium' if abs(cohens_d) > 0.5 else 'small'),
                'mean_class_0': mean_0,
                'mean_class_1': mean_1,
                'presence_class_0': presence_0,
                'presence_class_1': presence_1,
                'presence_diff': presence_diff,
                'discriminative': p_value < 0.05 and abs(cohens_d) > 0.5
            })

        df = pd.DataFrame(results).sort_values('p_value')
        return df

    # ========== Visualization ==========

    def plot_heatmap(self, top_k=20, save_path=None):
        """Heatmap of top-k graphlet presence by class."""
        sig_df = self.test_significance()

        if len(sig_df) == 0:
            print("No significant graphlets found!")
            return

        top_k = min(top_k, len(sig_df))
        top_graphlets = sig_df.head(top_k)['graphlet_id'].values

        # Compute presence rate
        num_classes = len(np.unique(self.labels))
        presence_matrix = np.zeros((len(top_graphlets), num_classes))

        for i, graphlet_idx in enumerate(top_graphlets):
            for c in range(num_classes):
                class_mask = self.labels == c
                counts = self.graphlet_matrix[class_mask, graphlet_idx]
                presence_matrix[i, c] = np.mean(counts > 0)

        # Plot
        plt.figure(figsize=(8, max(6, top_k * 0.3)))
        sns.heatmap(presence_matrix,
                    xticklabels=[f'Class {c}' for c in range(num_classes)],
                    yticklabels=[f'g_{idx}' for idx in top_graphlets],
                    cmap='RdYlGn', annot=True, fmt='.2f',
                    cbar_kws={'label': 'Presence Rate'},
                    vmin=0, vmax=1)
        plt.title(f'Top {top_k} Discriminative Graphlets\n{self.dataset_name} (k={self.k})')
        plt.xlabel('Target Class')
        plt.ylabel('Graphlet ID')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Heatmap saved to {save_path}")

        plt.show()

    def plot_count_distributions(self, top_k=5, save_path=None):
        """Violin plots for top-k discriminative graphlets."""
        sig_df = self.test_significance()

        if len(sig_df) == 0:
            print("No significant graphlets found!")
            return

        top_k = min(top_k, len(sig_df))
        top_graphlets = sig_df.head(top_k)['graphlet_id'].values

        fig, axes = plt.subplots(1, top_k, figsize=(4*top_k, 4))
        if top_k == 1:
            axes = [axes]

        for i, graphlet_idx in enumerate(top_graphlets):
            counts = self.graphlet_matrix[:, graphlet_idx]

            data = pd.DataFrame({
                'Count': counts,
                'Class': [f'Class {int(label)}' for label in self.labels]
            })

            sns.violinplot(data=data, x='Class', y='Count', ax=axes[i], palette='Set2')

            # Add stats
            p_val = sig_df[sig_df['graphlet_id'] == graphlet_idx]['p_value'].values[0]
            cohens_d = sig_df[sig_df['graphlet_id'] == graphlet_idx]['cohens_d'].values[0]

            axes[i].set_title(f'Graphlet {graphlet_idx}\np={p_val:.4f}, d={cohens_d:.2f}')
            axes[i].set_ylabel('Count' if i == 0 else '')

        plt.suptitle(f'Top {top_k} Discriminative Graphlet Distributions\n{self.dataset_name} (k={self.k})',
                     fontsize=14, y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Distribution plot saved to {save_path}")

        plt.show()

    # ========== Level 4: Predictive Baseline ==========

    def evaluate_predictive_baselines(self, cv_folds=5):
        """
        Level 4: Train simple classifiers on graphlet counts.

        Shows what accuracy can be achieved using ONLY graphlet features.
        Useful for ablation: compares to SS-GNN to show value of learned representations.
        """
        print(f"\n{'='*80}")
        print("LEVEL 4: PREDICTIVE BASELINE ANALYSIS")
        print("="*80)
        print(f"Training simple classifiers on graphlet count features...")
        print(f"Cross-validation: {cv_folds} folds\n")

        X = self.graphlet_matrix  # [num_graphs, num_graphlet_types]
        y = self.labels

        # Check if binary or multi-class
        num_classes = len(np.unique(y))
        is_binary = num_classes == 2

        print(f"Task: {'Binary' if is_binary else f'{num_classes}-class'} classification")
        print(f"Feature matrix: {X.shape}")
        print(f"Class distribution: {np.bincount(y.astype(int))}\n")

        # Define classifiers
        classifiers = {
            'Logistic Regression': Pipeline([
                ('scaler', StandardScaler()),
                ('clf', LogisticRegression(max_iter=1000, random_state=42))
            ]),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            ),
            'Linear SVM': Pipeline([
                ('scaler', StandardScaler()),
                ('clf', SVC(kernel='linear', random_state=42))
            ]),
        }

        # Cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        results = {}

        for name, clf in classifiers.items():
            print(f"Training {name}...")
            try:
                scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy', n_jobs=-1)

                results[name] = {
                    'mean_accuracy': scores.mean(),
                    'std_accuracy': scores.std(),
                    'scores': scores
                }

                print(f"  Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
                print(f"  Scores: {scores}")

                # Get feature importance (for Random Forest)
                if name == 'Random Forest':
                    clf.fit(X, y)
                    importances = clf.feature_importances_
                    top_k = 10
                    top_indices = np.argsort(importances)[-top_k:][::-1]
                    results[name]['top_features'] = top_indices.tolist()
                    results[name]['top_importances'] = importances[top_indices].tolist()

                    print(f"  Top 10 important graphlets: {top_indices.tolist()}")

            except Exception as e:
                print(f"  Error: {e}")
                results[name] = {'error': str(e)}

            print()

        # Compare to random baseline
        random_baseline = 1.0 / num_classes
        print(f"Random baseline: {random_baseline:.3f}")
        print()

        return results

    def compare_to_significance(self, predictive_results):
        """
        Compare Random Forest feature importance to statistical significance.

        Shows if model learns what's statistically important.
        """
        if 'Random Forest' not in predictive_results:
            return None

        rf_results = predictive_results['Random Forest']
        if 'top_features' not in rf_results:
            return None

        # Get statistical significance ranking
        sig_df = self.test_significance()

        # Compare top features
        rf_top = set(rf_results['top_features'][:10])
        stat_top = set(sig_df.head(10)['graphlet_id'].values)

        overlap = len(rf_top & stat_top)

        print(f"\n{'='*80}")
        print("COMPARISON: Statistical Significance vs. Model Importance")
        print("="*80)
        print(f"Top 10 by Random Forest importance: {sorted(rf_top)}")
        print(f"Top 10 by statistical significance: {sorted(stat_top)}")
        print(f"Overlap: {overlap}/10 ({overlap*10}%)")

        if overlap >= 7:
            print("✓ HIGH agreement - Model learns statistically important features")
        elif overlap >= 4:
            print("⚠ MODERATE agreement - Some alignment between stats and learning")
        else:
            print("✗ LOW agreement - Model may learn different patterns")

        return {
            'rf_top': [int(x) for x in rf_top],
            'stat_top': [int(x) for x in stat_top],
            'overlap': int(overlap)
        }

    # ========== Report ==========

    def generate_report(self, output_dir=None, include_predictive=True):
        """Generate comprehensive Level 1 analysis report."""
        print("\n" + "="*80)
        print("LEVEL 1: INDIVIDUAL GRAPHLET TYPE ANALYSIS")
        print("="*80)

        print(f"\nDataset: {self.dataset_name}")
        print(f"Graphs: {len(self.labels)}")
        print(f"Graphlet types: {len(self.wl_vocab)}")
        print(f"k-graphlet size: {self.k}")
        print(f"Samples per graph: {self.m}")
        print(f"Classes: {np.unique(self.labels)}")

        # Statistical tests
        sig_df = self.test_significance()
        num_discriminative = sig_df['discriminative'].sum()

        print(f"\n{'='*80}")
        print("STATISTICAL SIGNIFICANCE")
        print("="*80)
        print(f"Discriminative graphlets (p<0.05, |d|>0.5): {num_discriminative}/{len(sig_df)}")

        if num_discriminative > 0:
            print(f"\n{'='*80}")
            print("TOP 10 DISCRIMINATIVE GRAPHLETS")
            print("="*80)
            print(sig_df[['graphlet_id', 'p_value', 'cohens_d', 'effect_size',
                          'mean_class_0', 'mean_class_1', 'presence_diff']].head(10).to_string(index=False))

            # Interpretation
            print(f"\n{'='*80}")
            print("INTERPRETATION")
            print("="*80)

            avg_effect_size = sig_df.head(10)['cohens_d'].abs().mean()
            avg_p_value = sig_df.head(10)['p_value'].mean()

            if num_discriminative > 10 and avg_effect_size > 1.0:
                print("✓ STRONG SIGNAL: Many graphlets with large effect sizes")
                print("  → Dataset has clear discriminative k-graphlet structures")
                print("  → SS-GNN should work well with current k")
            elif num_discriminative > 5 and avg_effect_size > 0.5:
                print("⚠ MODERATE SIGNAL: Some graphlets show moderate differences")
                print("  → Dataset has some discriminative structure")
                print("  → SS-GNN may work with hyperparameter tuning")
            else:
                print("✗ WEAK SIGNAL: Few graphlets differ significantly")
                print("  → Current k may not capture discriminative structures")
                print("  → Try different k values (k±1, k±2)")
        else:
            print("\n✗ NO DISCRIMINATIVE GRAPHLETS FOUND")
            print("  → Current k does not capture class differences")
            print("  → Recommendations:")
            print("    1. Try different k values")
            print("    2. Check if dataset has structural differences")
            print("    3. Consider different feature encoding")

        # Save results
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

            # Save significance table
            sig_path = os.path.join(output_dir, 'significance_analysis.csv')
            sig_df.to_csv(sig_path, index=False)
            print(f"\n✓ Significance analysis saved to {sig_path}")

            # Save statistics table
            stats_df = self.compute_statistics()
            stats_path = os.path.join(output_dir, 'graphlet_statistics.csv')
            stats_df.to_csv(stats_path, index=False)
            print(f"✓ Graphlet statistics saved to {stats_path}")

            # Save visualizations
            heatmap_path = os.path.join(output_dir, 'heatmap.png')
            self.plot_heatmap(top_k=20, save_path=heatmap_path)

            dist_path = os.path.join(output_dir, 'distributions.png')
            self.plot_count_distributions(top_k=5, save_path=dist_path)

        # Level 4: Predictive Baseline
        predictive_results = None
        if include_predictive:
            predictive_results = self.evaluate_predictive_baselines(cv_folds=5)

            # Compare to statistical significance
            comparison = self.compare_to_significance(predictive_results)

            # Save predictive results
            if output_dir:
                pred_results_data = []
                for name, metrics in predictive_results.items():
                    if 'error' not in metrics:
                        pred_results_data.append({
                            'classifier': name,
                            'mean_accuracy': metrics['mean_accuracy'],
                            'std_accuracy': metrics['std_accuracy']
                        })

                pred_df = pd.DataFrame(pred_results_data)
                pred_path = os.path.join(output_dir, 'predictive_baselines.csv')
                pred_df.to_csv(pred_path, index=False)
                print(f"\n✓ Predictive baseline results saved to {pred_path}")

                # Save comparison
                if comparison:
                    comp_path = os.path.join(output_dir, 'significance_vs_importance.json')
                    import json
                    with open(comp_path, 'w') as f:
                        json.dump(comparison, f, indent=2)
                    print(f"✓ Significance comparison saved to {comp_path}")

        print(f"\n{'='*80}")
        print("END OF ANALYSIS")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Graphlet-Label Correlation Analysis (Level 1)')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (e.g., MUTAG, ZINC, CSL)')
    parser.add_argument('--k', type=int, default=6,
                        help='Subgraph size (default: 6)')
    parser.add_argument('--m', type=int, default=1000,
                        help='Number of samples per graph (default: 1000)')
    parser.add_argument('--sampler', type=str, default='ugs',
                        choices=['ugs', 'uniform', 'rwr'],
                        help='Sampler to use (default: ugs)')
    parser.add_argument('--use-features', action='store_true',
                        help='Use node features in WL hash (default: False)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for results (default: None)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--skip-predictive', action='store_true',
                        help='Skip Level 4 predictive baseline analysis (default: False)')

    args = parser.parse_args()

    # Create analyzer
    analyzer = GraphletAnalyzer(
        dataset_name=args.dataset,
        k=args.k,
        m=args.m,
        sampler_name=args.sampler,
        use_node_features=args.use_features,
        seed=args.seed
    )

    # Generate report
    output_dir = args.output or f"experiments/graphlet_analysis/{args.dataset}_k{args.k}"
    analyzer.generate_report(output_dir=output_dir, include_predictive=not args.skip_predictive)


if __name__ == '__main__':
    main()
