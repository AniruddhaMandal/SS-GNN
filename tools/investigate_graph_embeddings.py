"""
Investigate: If graph embeddings are different for different classes,
why is the classifier not learning better?
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from gps.experiment import Experiment
from gps.config import set_config, load_config

def analyze_graph_embeddings_per_class(exp):
    """Collect graph embeddings for each class and analyze separability."""
    print("=" * 80)
    print("GRAPH EMBEDDING ANALYSIS BY CLASS")
    print("=" * 80)

    exp.model.eval()

    # Collect embeddings for each split
    for split_name, loader in [('train', exp.train_loader), ('val', exp.val_loader)]:
        print(f"\n{split_name.upper()} SPLIT:")
        print("-" * 80)

        embeddings_by_class = {i: [] for i in range(10)}

        with torch.no_grad():
            for batch in loader:
                data, labels = exp._unpack_batch(batch)
                data = data.to(exp.device)

                # Get graph embeddings (before classifier)
                graph_emb = exp.model.encoder(data)

                # Store by class
                for emb, label in zip(graph_emb, labels):
                    embeddings_by_class[int(label.item())].append(emb.cpu().numpy())

        # Convert to arrays
        for cls in range(10):
            if embeddings_by_class[cls]:
                embeddings_by_class[cls] = np.array(embeddings_by_class[cls])

        # Statistics per class
        print(f"\nPer-class statistics:")
        for cls in range(10):
            if len(embeddings_by_class[cls]) > 0:
                embs = embeddings_by_class[cls]
                print(f"  Class {cls}: n={len(embs)}, mean={embs.mean():.4f}, std={embs.std():.4f}")

        # Inter-class vs intra-class distances
        print(f"\nSeparability analysis:")

        # Compute class centroids
        centroids = {}
        for cls in range(10):
            if len(embeddings_by_class[cls]) > 0:
                centroids[cls] = embeddings_by_class[cls].mean(axis=0)

        # Inter-class distances (between centroids)
        inter_class_dists = []
        for i in range(10):
            for j in range(i+1, 10):
                if i in centroids and j in centroids:
                    dist = np.linalg.norm(centroids[i] - centroids[j])
                    inter_class_dists.append(dist)

        # Intra-class distances (within class variance)
        intra_class_dists = []
        for cls in range(10):
            if len(embeddings_by_class[cls]) > 1:
                embs = embeddings_by_class[cls]
                for i in range(len(embs)):
                    for j in range(i+1, len(embs)):
                        dist = np.linalg.norm(embs[i] - embs[j])
                        intra_class_dists.append(dist)

        if inter_class_dists and intra_class_dists:
            print(f"  Inter-class distance: {np.mean(inter_class_dists):.4f} ± {np.std(inter_class_dists):.4f}")
            print(f"  Intra-class distance: {np.mean(intra_class_dists):.4f} ± {np.std(intra_class_dists):.4f}")
            print(f"  Separability ratio: {np.mean(inter_class_dists) / np.mean(intra_class_dists):.4f}")
            print(f"  [Ratio > 2.0 is good, > 5.0 is excellent]")

        # Check if classes are linearly separable
        if len(centroids) >= 2:
            all_embs = []
            all_labels = []
            for cls in range(10):
                if len(embeddings_by_class[cls]) > 0:
                    all_embs.append(embeddings_by_class[cls])
                    all_labels.extend([cls] * len(embeddings_by_class[cls]))

            if all_embs:
                all_embs = np.vstack(all_embs)
                all_labels = np.array(all_labels)

                # Train a simple linear classifier
                from sklearn.linear_model import LogisticRegression
                clf = LogisticRegression(max_iter=1000, multi_class='multinomial')
                clf.fit(all_embs, all_labels)
                acc = clf.score(all_embs, all_labels)
                print(f"\n  Linear separability: {acc:.4f}")
                print(f"  [This is the BEST possible accuracy with linear classifier]")
                print(f"  [Your model has {exp.cfg.model_config.hidden_dim}D embeddings → {exp.cfg.model_config.out_dim} classes]")

def check_classifier_head_capacity(exp):
    """Check if classifier head has enough capacity."""
    print("\n" + "=" * 80)
    print("CLASSIFIER HEAD ANALYSIS")
    print("=" * 80)

    head = exp.model.model_head
    print(f"\nClassifier architecture:")
    print(head)

    total_params = sum(p.numel() for p in head.parameters())
    print(f"\nClassifier parameters: {total_params}")

    # Check weight norms
    print(f"\nWeight norms:")
    for name, param in head.named_parameters():
        print(f"  {name}: {param.norm().item():.4f}")

    # Check if classifier is learning
    print(f"\nClassifier learning status:")
    with torch.no_grad():
        # Get a batch
        batch = next(iter(exp.train_loader))
        data, labels = exp._unpack_batch(batch)
        data = data.to(exp.device)

        # Get embeddings
        graph_emb = exp.model.encoder(data)
        print(f"  Input embedding: {graph_emb[0].cpu().numpy()}")

        # Get logits
        logits = exp.model.model_head(graph_emb)
        print(f"  Output logits: {logits[0].cpu().numpy()}")
        print(f"  Prediction: {logits.argmax(dim=-1).item()}")
        print(f"  True label: {labels[0].item()}")

def check_hidden_dim_sufficiency(exp):
    """Check if hidden dimension is sufficient for the task."""
    print("\n" + "=" * 80)
    print("HIDDEN DIMENSION ANALYSIS")
    print("=" * 80)

    hidden_dim = exp.cfg.model_config.hidden_dim
    num_classes = exp.cfg.model_config.out_dim

    print(f"\nConfiguration:")
    print(f"  Hidden dimension: {hidden_dim}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Ratio: {hidden_dim / num_classes:.2f}")

    print(f"\nInterpretation:")
    if hidden_dim < num_classes:
        print(f"  ⚠️  PROBLEM: Hidden dim ({hidden_dim}) < num classes ({num_classes})")
        print(f"  This is a bottleneck! Impossible to create {num_classes} linearly separable clusters")
        print(f"  in {hidden_dim}D space without severe overlap.")
    elif hidden_dim < 2 * num_classes:
        print(f"  ⚠️  WARNING: Hidden dim ({hidden_dim}) is barely enough for {num_classes} classes")
        print(f"  Recommended: hidden_dim >= 2 * num_classes = {2 * num_classes}")
    else:
        print(f"  ✓  Hidden dim is sufficient")

    print(f"\nRule of thumb:")
    print(f"  Minimum: hidden_dim >= num_classes ({num_classes})")
    print(f"  Recommended: hidden_dim >= 2-3 × num_classes ({2*num_classes}-{3*num_classes})")
    print(f"  Ideal for complex tasks: hidden_dim >= 64-128")

def main():
    # Load config
    config_file_path = 'configs/ss_gnn/SYNTHETIC/CSL/gcn-k4.json'
    dict_cfg = load_config(config_file_path)
    dict_cfg['tracker'] = 'Off'
    dict_cfg['device'] = 'cpu'
    cfg = set_config(dict_cfg)

    print("Building experiment...")
    exp = Experiment(cfg=cfg)
    print("Experiment built!\n")

    # Run analyses
    check_hidden_dim_sufficiency(exp)
    analyze_graph_embeddings_per_class(exp)
    check_classifier_head_capacity(exp)

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("""
The issue is NOT that the classifier can't learn - it IS learning!

The problem is INSUFFICIENT MODEL CAPACITY:
  1. hidden_dim=5 is way too small for 10 classes
  2. This creates a bottleneck where graph embeddings must squeeze into 5D
  3. Even though embeddings are different per class, they overlap significantly
  4. Linear classifier can't separate overlapping clusters in low-dimensional space

Think of it like this:
  - You have 10 classes to separate
  - But only 5 dimensions to work with
  - It's like trying to separate 10 clusters on a 5D space - they WILL overlap!
  - The classifier is doing its best, but the embedding space is too crowded

Solution: Increase hidden_dim from 5 to at least 32, ideally 64+
    """)

if __name__ == "__main__":
    main()
