"""
Build WL Vocabulary for SS-GNN-WL

This script builds a WL vocabulary from a dataset and saves it to disk.
The vocabulary must be built before training SS-GNN-WL models.

Usage:
    python tools/build_wl_vocab.py --dataset CSL --k 6 --m 100 --output wl_vocabs/csl_k6.pkl

Arguments:
    --dataset: Dataset name (CSL, MUTAG, PROTEINS, etc.)
    --k: Subgraph size
    --m: Number of subgraphs to sample per graph
    --output: Output path for vocabulary file
    --sampler: Sampler to use (uniform, ugs, rwr) - default: uniform
    --max-graphs: Maximum number of graphs to process (None = all)
    --wl-iterations: Number of WL iterations (default: 3)
    --use-node-features: Use node features in WL hash (default: False)
    --seed: Random seed (default: 42)
"""

import argparse
import sys
import os

# Add src/gps to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'gps'))

import torch
from gps.config import load_config, set_config
from gps.utils.wl_vocab import build_wl_vocabulary_from_loader, save_wl_vocabulary


def main():
    parser = argparse.ArgumentParser(description='Build WL vocabulary for SS-GNN-WL')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (CSL, MUTAG, PROTEINS, etc.)')
    parser.add_argument('--k', type=int, required=True,
                       help='Subgraph size')
    parser.add_argument('--m', type=int, default=100,
                       help='Number of subgraphs to sample per graph (default: 100)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for vocabulary file (e.g., wl_vocabs/csl_k6.pkl)')
    parser.add_argument('--sampler', type=str, default='uniform',
                       choices=['uniform', 'ugs', 'rwr', 'epsilon_uniform'],
                       help='Sampler to use (default: uniform)')
    parser.add_argument('--max-graphs', type=int, default=None,
                       help='Maximum number of graphs to process (default: all)')
    parser.add_argument('--wl-iterations', type=int, default=3,
                       help='Number of WL iterations (default: 3)')
    parser.add_argument('--use-node-features', action='store_true',
                       help='Use node features in WL hash')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')

    args = parser.parse_args()

    print("=" * 80)
    print("Building WL Vocabulary for SS-GNN-WL")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Subgraph size (k): {args.k}")
    print(f"Samples per graph (m): {args.m}")
    print(f"Sampler: {args.sampler}")
    print(f"WL iterations: {args.wl_iterations}")
    print(f"Use node features: {args.use_node_features}")
    print(f"Max graphs: {args.max_graphs if args.max_graphs else 'all'}")
    print(f"Output path: {args.output}")
    print("=" * 80)

    # Import the appropriate sampler
    if args.sampler == 'uniform':
        try:
            import uniform_sampler
            sampler = uniform_sampler
            print("✓ Using uniform sampler (exact)")
        except ImportError:
            print("✗ Error: uniform_sampler not installed")
            print("  Run: pip install -e src/samplers/uniform_sampler --no-build-isolation")
            sys.exit(1)
    elif args.sampler == 'ugs':
        try:
            import ugs_sampler
            sampler = ugs_sampler
            print("✓ Using UGS sampler (approximate)")
        except ImportError:
            print("✗ Error: ugs_sampler not installed")
            print("  Run: pip install -e src/samplers/ugs_sampler --no-build-isolation")
            sys.exit(1)
    elif args.sampler == 'rwr':
        try:
            import rwr_sampler
            sampler = rwr_sampler
            print("✓ Using RWR sampler (biased)")
        except ImportError:
            print("✗ Error: rwr_sampler not installed")
            print("  Run: pip install -e src/samplers/rwr_sampler --no-build-isolation")
            sys.exit(1)
    elif args.sampler == 'epsilon_uniform':
        try:
            import epsilon_uniform_sampler
            sampler = epsilon_uniform_sampler
            print("✓ Using epsilon-uniform sampler (fast approximate)")
        except ImportError:
            print("✗ Error: epsilon_uniform_sampler not installed")
            print("  Run: pip install -e src/samplers/epsilon_uniform_sampler --no-build-isolation")
            sys.exit(1)

    # Create a minimal config to load the dataset
    # We'll load the dataset using the registered dataset loader
    from gps.registry import get_dataset
    from gps import ExperimentConfig, ModelConfig, TrainConfig, SubgraphParam

    cfg = ExperimentConfig(
        name=f"vocab_builder_{args.dataset}",
        dataset_name=args.dataset,
        model_name="VANILLA",  # Dummy model (not used)
        task="Binary-Classification",  # Dummy task
        model_config=ModelConfig(
            node_feature_dim=1,
            edge_feature_dim=1,
            hidden_dim=64,
            out_dim=2,
            subgraph_sampling=True,
            subgraph_param=SubgraphParam(k=args.k, m=args.m, pooling="mean"),
            kwargs={'node_feature_type': 'lap_pe'}  # Required for some datasets (CSL, etc.)
        ),
        train=TrainConfig(
            train_batch_size=32,
            val_batch_size=32,
            epochs=1
        ),
        cache_dir='cache',  # Required for dataset caching
        seed=args.seed
    )

    # Build dataloaders
    print(f"\nLoading dataset: {args.dataset}")
    try:
        dataloader_fn = get_dataset(args.dataset)
        train_loader, val_loader, test_loader = dataloader_fn(cfg)
        print(f"✓ Dataset loaded: {len(train_loader.dataset)} training graphs")
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        sys.exit(1)

    # Build vocabulary from training data
    wl_vocab = build_wl_vocabulary_from_loader(
        dataloader=train_loader,
        sampler=sampler,
        k=args.k,
        m=args.m,
        num_iterations=args.wl_iterations,
        max_graphs=args.max_graphs,
        use_node_features=args.use_node_features,
        seed=args.seed
    )

    # Save vocabulary
    save_wl_vocabulary(wl_vocab, args.output)

    print("\n" + "=" * 80)
    print("Vocabulary Statistics")
    print("=" * 80)
    print(f"Vocabulary size: {len(wl_vocab)} unique WL-equivalence classes")
    print(f"Recommended WL embedding dimension: {max(64, int(2 * len(wl_vocab) ** 0.5))}")
    print(f"Saved to: {args.output}")
    print("=" * 80)
    print("\n✓ Done! You can now use this vocabulary with SS-GNN-WL.")
    print(f"\nExample config entry:")
    print('  "model_config": {')
    print('    ...')
    print('    "kwargs": {')
    print(f'      "wl_vocab_path": "{args.output}",')
    print('      "wl_dim": 64,')
    print('      "use_node_features_in_wl": false,')
    print('      "wl_iterations": 3')
    print('    }')
    print('  }')


if __name__ == '__main__':
    main()
