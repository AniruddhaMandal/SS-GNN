"""
Test script for MolHIV dataset.
Downloads the dataset and prints details.
"""

import sys
sys.path.insert(0, 'src/gps')

from gps.dataset_loaders.molhiv import MolHIVDataset
from gps.encoder import OGBAtomEncoder, OGBBondEncoder
from torch_geometric.transforms import Compose


def main():
    print("=" * 60)
    print("MolHIV Dataset Test")
    print("=" * 60)

    # Create transforms
    emb_dim = 64
    transforms = Compose([
        OGBAtomEncoder(emb_dim=emb_dim),
        OGBBondEncoder(emb_dim=emb_dim),
    ])

    # Load dataset (will download if not present)
    print("\nLoading dataset (downloading if needed)...")
    dataset = MolHIVDataset(root='./data/OGB/molhiv', transform=transforms)

    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    print(f"Total graphs:        {len(dataset)}")
    print(f"Num node features:   {dataset.num_node_features} (categorical) -> {emb_dim} (embedded)")
    print(f"Num edge features:   {dataset.num_edge_features} (categorical) -> {emb_dim} (embedded)")
    print(f"Num classes:         {dataset.num_classes}")

    # Get splits
    split = dataset.get_idx_split()
    print(f"\nSplit sizes:")
    print(f"  Train: {len(split['train'])} ({100*len(split['train'])/len(dataset):.1f}%)")
    print(f"  Valid: {len(split['valid'])} ({100*len(split['valid'])/len(dataset):.1f}%)")
    print(f"  Test:  {len(split['test'])} ({100*len(split['test'])/len(dataset):.1f}%)")

    # Class distribution
    print("\nClass distribution:")
    labels = [dataset[i].y.item() for i in range(len(dataset))]
    num_pos = sum(labels)
    num_neg = len(labels) - num_pos
    print(f"  Negative (0): {num_neg} ({100*num_neg/len(dataset):.1f}%)")
    print(f"  Positive (1): {num_pos} ({100*num_pos/len(dataset):.1f}%)")

    # Graph statistics
    print("\nGraph statistics:")
    num_nodes = [dataset[i].x.shape[0] for i in range(min(1000, len(dataset)))]
    num_edges = [dataset[i].edge_index.shape[1] for i in range(min(1000, len(dataset)))]
    print(f"  Avg nodes per graph: {sum(num_nodes)/len(num_nodes):.1f}")
    print(f"  Avg edges per graph: {sum(num_edges)/len(num_edges):.1f}")
    print(f"  Min/Max nodes: {min(num_nodes)} / {max(num_nodes)}")
    print(f"  Min/Max edges: {min(num_edges)} / {max(num_edges)}")

    # Sample data
    print("\n" + "=" * 60)
    print("Sample Graphs")
    print("=" * 60)
    for i in [0, 1, 2]:
        data = dataset[i]
        print(f"\nGraph {i}:")
        print(f"  SMILES: {data.smiles}")
        print(f"  Label:  {data.y.item()} ({'HIV active' if data.y.item() == 1 else 'HIV inactive'})")
        print(f"  Nodes:  {data.x.shape[0]}")
        print(f"  Edges:  {data.edge_index.shape[1]}")
        print(f"  x shape: {data.x.shape}")
        print(f"  edge_attr shape: {data.edge_attr.shape}")

    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
