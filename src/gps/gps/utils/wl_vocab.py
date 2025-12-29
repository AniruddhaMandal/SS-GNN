"""
WL Vocabulary Builder for SS-GNN-WL

This module provides utilities for:
1. Computing WL hashes for PyTorch Geometric graphs
2. Building WL vocabulary from datasets
3. Saving/loading vocabulary for reproducibility
"""

import torch
import networkx as nx
import pickle
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, Optional, Tuple
from tqdm import tqdm
import hashlib


def compute_wl_hash(edge_index: torch.Tensor,
                   num_nodes: int,
                   node_features: Optional[torch.Tensor] = None,
                   num_iterations: int = 3) -> str:
    """
    Compute WL hash for a subgraph represented as PyG format.

    Args:
        edge_index: [2, num_edges] edge connectivity
        num_nodes: number of nodes in subgraph
        node_features: [num_nodes, feature_dim] optional node features
        num_iterations: number of WL iterations (default: 3)

    Returns:
        WL hash string
    """
    # Convert PyG graph to NetworkX
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))

    if edge_index.numel() > 0:
        edges = edge_index.t().cpu().numpy()
        G.add_edges_from(edges)

    # Add node attributes if provided
    if node_features is not None:
        # Hash node features to create categorical attributes
        for i in range(num_nodes):
            feat_hash = hashlib.md5(node_features[i].cpu().numpy().tobytes()).hexdigest()[:8]
            G.nodes[i]['attr'] = feat_hash
    else:
        # Use degree as node attribute if no features
        for i in range(num_nodes):
            G.nodes[i]['attr'] = str(G.degree(i))

    # Compute WL hash
    try:
        wl_hash = nx.weisfeiler_lehman_graph_hash(
            G,
            node_attr='attr',
            iterations=num_iterations
        )
    except:
        # Fallback: if WL fails, use simple graph hash
        wl_hash = f"deg_{sum(dict(G.degree()).values())}_edges_{G.number_of_edges()}"

    return wl_hash


def extract_subgraph_from_batch(batch,
                                 subgraph_idx: int,
                                 nodes_t: torch.Tensor,
                                 edge_index_t: torch.Tensor,
                                 edge_ptr_t: torch.Tensor) -> Tuple[torch.Tensor, int, Optional[torch.Tensor]]:
    """
    Extract a single subgraph from batched subgraph data.

    Args:
        batch: SubgraphFeaturesBatch
        subgraph_idx: index of subgraph to extract
        nodes_t: [num_samples, k] global node IDs
        edge_index_t: [2, total_edges] local edge indices
        edge_ptr_t: [num_samples+1] edge offsets

    Returns:
        edge_index: [2, num_edges] for this subgraph
        num_nodes: k (subgraph size)
        node_features: [k, feature_dim] or None
    """
    # Get nodes for this subgraph
    subgraph_nodes = nodes_t[subgraph_idx]  # [k]
    valid_mask = (subgraph_nodes >= 0)
    num_nodes = valid_mask.sum().item()

    # Get edges for this subgraph
    edge_start = edge_ptr_t[subgraph_idx]
    edge_end = edge_ptr_t[subgraph_idx + 1]
    subgraph_edges = edge_index_t[:, edge_start:edge_end]  # [2, num_edges]

    # Get node features (optional)
    if batch.x is not None and num_nodes > 0:
        valid_nodes = subgraph_nodes[valid_mask]
        node_features = batch.x[valid_nodes]  # [num_nodes, feature_dim]
    else:
        node_features = None

    return subgraph_edges, num_nodes, node_features


def build_wl_vocabulary_from_loader(dataloader,
                                   sampler,
                                   k: int,
                                   m: int,
                                   num_iterations: int = 3,
                                   max_graphs: Optional[int] = None,
                                   use_node_features: bool = False,
                                   seed: int = 42) -> Dict[str, int]:
    """
    Build WL vocabulary from a PyG dataloader.

    Args:
        dataloader: PyTorch Geometric DataLoader
        sampler: Sampler function (e.g., uniform_sampler.sample_batch)
        k: subgraph size
        m: number of subgraphs to sample per graph
        num_iterations: WL iterations
        max_graphs: maximum number of graphs to process (None = all)
        use_node_features: whether to use node features in WL hash
        seed: random seed

    Returns:
        Dictionary mapping WL hash -> unique ID
    """
    wl_vocab = {}
    current_id = 0

    graphs_processed = 0

    print(f"Building WL vocabulary (k={k}, m={m}, WL_iter={num_iterations})...")

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
        # Sample subgraphs
        import numpy as np
        nodes_t, edge_index_t, edge_ptr_t, sample_ptr_t, edge_src_t = sampler.sample_batch(
            edge_index=batch.edge_index,
            ptr=batch.ptr,
            m_per_graph=m,
            k=k,
            mode="sample",
            seed=seed + batch_idx
        )

        # Process each subgraph
        num_subgraphs = nodes_t.shape[0]

        for i in range(num_subgraphs):
            subgraph_edges, num_nodes, node_features = extract_subgraph_from_batch(
                batch, i, nodes_t, edge_index_t, edge_ptr_t
            )

            if num_nodes == 0:
                continue

            # Compute WL hash
            wl_hash = compute_wl_hash(
                edge_index=subgraph_edges,
                num_nodes=num_nodes,
                node_features=node_features if use_node_features else None,
                num_iterations=num_iterations
            )

            # Add to vocabulary if new
            if wl_hash not in wl_vocab:
                wl_vocab[wl_hash] = current_id
                current_id += 1

        graphs_processed += batch.num_graphs

        if max_graphs is not None and graphs_processed >= max_graphs:
            break

    print(f"✓ Vocabulary built: {len(wl_vocab)} unique WL-equivalence classes")
    return wl_vocab


def save_wl_vocabulary(wl_vocab: Dict[str, int], save_path: str):
    """Save WL vocabulary to disk."""
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    with open(save_path, 'wb') as f:
        pickle.dump(wl_vocab, f)
    print(f"✓ Vocabulary saved to {save_path}")


def load_wl_vocabulary(load_path: str) -> Dict[str, int]:
    """Load WL vocabulary from disk."""
    with open(load_path, 'rb') as f:
        wl_vocab = pickle.load(f)
    print(f"✓ Vocabulary loaded from {load_path} ({len(wl_vocab)} classes)")
    return wl_vocab


def hash_to_id(wl_hash: str, wl_vocab: Dict[str, int]) -> int:
    """
    Convert WL hash to integer ID.

    Args:
        wl_hash: WL hash string
        wl_vocab: vocabulary dictionary

    Returns:
        Integer ID (uses len(wl_vocab) for unknown hashes)
    """
    return wl_vocab.get(wl_hash, len(wl_vocab))  # unknown -> last ID


def get_vocab_stats(wl_vocab: Dict[str, int]) -> Dict:
    """Get statistics about the vocabulary."""
    return {
        'vocab_size': len(wl_vocab),
        'recommended_embedding_dim': max(64, int(2 * len(wl_vocab) ** 0.5))
    }
