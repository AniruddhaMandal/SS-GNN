"""
SS-GNN-WL: Subgraph Sampling GNN with Frozen WL Embeddings

Architecture: Frozen WL ⊕ Learnable GNN
- Frozen WL embeddings: Guarantee separation between WL-equivalence classes (theory)
- Learnable GNN: Task-specific continuous representations (practice)
- Concatenation: Best of both worlds
- MEAN aggregation: Sample complexity theory applies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from gps import SubgraphFeaturesBatch
from gps.models.ss_gnn import SubgraphGNNEncoder
from gps.utils.wl_vocab import compute_wl_hash, hash_to_id, extract_subgraph_from_batch
from typing import Dict, Optional


class FrozenWLEmbedding(nn.Module):
    """
    Frozen WL embedding layer that guarantees separation between WL-equivalence classes.

    - Initialized with orthogonal weights (approximately preserves distances)
    - Weights are frozen (requires_grad=False)
    - Maps WL hash -> fixed embedding vector
    """

    def __init__(self, vocab_size: int, embed_dim: int = 64):
        """
        Args:
            vocab_size: number of unique WL-equivalence classes
            embed_dim: embedding dimension (should be >= O(log(vocab_size)))
        """
        super().__init__()
        self.vocab_size = vocab_size + 1  # +1 for unknown hashes
        self.embed_dim = embed_dim

        # Create embedding layer
        self.embedding = nn.Embedding(self.vocab_size, embed_dim)

        # Initialize orthogonally (approximately preserves unit distances)
        # For embedding layers, we initialize the weight matrix orthogonally
        nn.init.orthogonal_(self.embedding.weight)

        # FREEZE - this is critical for theory!
        self.embedding.weight.requires_grad = False

    def forward(self, wl_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            wl_ids: [num_subgraphs] integer IDs from WL vocabulary

        Returns:
            embeddings: [num_subgraphs, embed_dim]
        """
        return self.embedding(wl_ids)


class SubgraphSamplingGNNWithWL(nn.Module):
    """
    SS-GNN-WL: Graph classifier with frozen WL embeddings + learnable GNN.

    Architecture:
        Subgraphs → [Frozen WL || GNN] → MEAN → Graph Embedding
                     Theory     Practice   Theory

    Args:
        in_channels: input node feature dimension
        edge_dim: edge feature dimension
        hidden_dim: GNN hidden dimension
        wl_vocab: WL vocabulary dictionary (hash -> id)
        wl_dim: frozen WL embedding dimension (default: 64)
        num_layers: number of GNN layers
        mlp_layers: number of MLP layers in GNN
        dropout: dropout rate
        conv_type: GNN type ('gin', 'gcn', 'gine', etc.)
        pooling: subgraph pooling ('mean', 'sum', 'max')
        use_node_features_in_wl: whether to use node features in WL hash
        wl_iterations: number of WL refinement iterations
    """

    def __init__(self,
                 in_channels: int,
                 edge_dim: int,
                 hidden_dim: int,
                 wl_vocab: Dict[str, int],
                 wl_dim: int = 64,
                 num_layers: int = 5,
                 mlp_layers: int = 2,
                 dropout: float = 0.1,
                 conv_type: str = 'gin',
                 pooling: str = 'mean',
                 use_node_features_in_wl: bool = False,
                 wl_iterations: int = 3):
        super().__init__()

        self.wl_vocab = wl_vocab
        self.wl_dim = wl_dim
        self.hidden_dim = hidden_dim
        self.use_node_features_in_wl = use_node_features_in_wl
        self.wl_iterations = wl_iterations

        vocab_size = len(wl_vocab)
        print(f"[SS-GNN-WL] Initializing with vocab_size={vocab_size}, wl_dim={wl_dim}, gnn_dim={hidden_dim}")

        # ============ FROZEN WL EMBEDDINGS (Separation Guarantee) ============
        self.wl_embedding = FrozenWLEmbedding(vocab_size, wl_dim)

        # ============ LEARNABLE GNN ENCODER (Task-Specific Learning) ============
        self.gnn_encoder = SubgraphGNNEncoder(
            in_channels=in_channels,
            out_channels=hidden_dim,
            hidden_dim=hidden_dim,
            edge_dim=edge_dim,
            num_layers=num_layers,
            conv_type=conv_type,
            mlp_layers=mlp_layers,
            dropout=dropout,
            pooling=pooling,
            res_connect=True,
            batch_norm=True
        )

        # Combined embedding dimension
        self.combined_dim = wl_dim + hidden_dim

        print(f"[SS-GNN-WL] Combined embedding dimension: {self.combined_dim} (WL:{wl_dim} + GNN:{hidden_dim})")

    def forward(self, batch: SubgraphFeaturesBatch) -> torch.Tensor:
        """
        Forward pass: Encode subgraphs with frozen WL + GNN, aggregate to graph level.

        Args:
            batch: SubgraphFeaturesBatch containing sampled subgraphs

        Returns:
            graph_embeddings: [num_graphs, wl_dim + hidden_dim]
        """
        x_global = batch.x
        edge_attr = batch.edge_attr
        nodes_t = batch.nodes_sampled
        edge_index_t = batch.edge_index_sampled
        edge_ptr_t = batch.edge_ptr
        sample_ptr_t = batch.sample_ptr
        edge_src_global_t = batch.edge_src_global

        device = x_global.device
        num_graphs = sample_ptr_t.size(0) - 1
        num_subgraphs = nodes_t.shape[0]

        # ============ STEP 1: Compute Frozen WL Embeddings ============
        wl_ids = self._compute_wl_ids(
            batch=batch,
            nodes_t=nodes_t,
            edge_index_t=edge_index_t,
            edge_ptr_t=edge_ptr_t
        )  # [num_subgraphs]

        wl_embeddings = self.wl_embedding(wl_ids)  # [num_subgraphs, wl_dim]

        # ============ STEP 2: Compute GNN Embeddings ============
        gnn_embeddings = self.encode_subgraphs_with_gnn(
            x=x_global,
            edge_attr=edge_attr,
            nodes_t=nodes_t,
            edge_index_t=edge_index_t,
            edge_ptr_t=edge_ptr_t,
            edge_src_global_t=edge_src_global_t
        )  # [num_subgraphs, hidden_dim]

        # ============ STEP 3: Concatenate Frozen WL + GNN ============
        combined_embeddings = torch.cat([wl_embeddings, gnn_embeddings], dim=-1)  # [num_subgraphs, wl_dim + hidden_dim]

        # ============ STEP 4: MEAN Aggregation (Theory Applies Here!) ============
        # Handle graphs with no valid subgraphs
        samples_per_graph = sample_ptr_t[1:] - sample_ptr_t[:-1]

        if (samples_per_graph == 0).any():
            # Some graphs have 0 valid subgraphs - add placeholder zero embeddings
            zero_emb = torch.zeros(1, combined_embeddings.size(1), device=device)
            new_sample_embs = []
            new_graph_ids = []

            for g in range(num_graphs):
                num_samples = samples_per_graph[g].item()
                if num_samples > 0:
                    start_idx = sample_ptr_t[g]
                    end_idx = sample_ptr_t[g + 1]
                    new_sample_embs.append(combined_embeddings[start_idx:end_idx])
                    new_graph_ids.extend([g] * num_samples)
                else:
                    new_sample_embs.append(zero_emb)
                    new_graph_ids.append(g)

            combined_embeddings = torch.cat(new_sample_embs, dim=0)
            global_graph_ptr = torch.tensor(new_graph_ids, dtype=torch.long, device=device)
        else:
            global_graph_ptr = torch.repeat_interleave(
                torch.arange(0, num_graphs, device=device),
                samples_per_graph
            )

        # MEAN aggregation (sample complexity theory applies!)
        graph_embeddings = global_mean_pool(combined_embeddings, global_graph_ptr)  # [num_graphs, combined_dim]

        return graph_embeddings

    def _compute_wl_ids(self,
                       batch: SubgraphFeaturesBatch,
                       nodes_t: torch.Tensor,
                       edge_index_t: torch.Tensor,
                       edge_ptr_t: torch.Tensor) -> torch.Tensor:
        """
        Compute WL IDs for all subgraphs in the batch.

        Returns:
            wl_ids: [num_subgraphs] integer IDs
        """
        num_subgraphs = nodes_t.shape[0]
        wl_ids = []

        for i in range(num_subgraphs):
            # Extract subgraph
            subgraph_edges, num_nodes, node_features = extract_subgraph_from_batch(
                batch, i, nodes_t, edge_index_t, edge_ptr_t
            )

            if num_nodes == 0:
                # Empty subgraph -> use unknown ID
                wl_ids.append(len(self.wl_vocab))
                continue

            # Compute WL hash
            wl_hash = compute_wl_hash(
                edge_index=subgraph_edges,
                num_nodes=num_nodes,
                node_features=node_features if self.use_node_features_in_wl else None,
                num_iterations=self.wl_iterations
            )

            # Convert to ID
            wl_id = hash_to_id(wl_hash, self.wl_vocab)
            wl_ids.append(wl_id)

        return torch.tensor(wl_ids, dtype=torch.long, device=nodes_t.device)

    def encode_subgraphs_with_gnn(self,
                                  x,
                                  edge_attr,
                                  nodes_t,
                                  edge_index_t,
                                  edge_ptr_t,
                                  edge_src_global_t):
        """
        Encode subgraphs using the learnable GNN encoder.
        (Reuses logic from SubgraphSamplingGNNClassifier)

        Returns:
            subgraph_embeddings: [num_subgraphs, hidden_dim]
        """
        device = x.device
        num_subgraphs, k = nodes_t.shape

        # Gather global node attributes
        stacked_nodes = nodes_t.flatten()

        # Handle -1 padding: clamp to 0 and create mask
        valid_mask = (stacked_nodes >= 0)
        stacked_nodes_clamped = stacked_nodes.clamp(min=0)

        global_x = x[stacked_nodes_clamped]

        # Zero out features for padded positions
        if not valid_mask.all():
            global_x = global_x * valid_mask.unsqueeze(-1).float()

        # Gather global edge attributes
        if edge_attr is not None:
            global_edge_attr = edge_attr[edge_src_global_t]
        else:
            global_edge_attr = None

        # Convert edge index to (subgraph)batch level
        global_edge_index = torch.repeat_interleave(
            torch.arange(0, num_subgraphs, device=device),
            edge_ptr_t[1:] - edge_ptr_t[:-1]
        ) * k
        global_edge_index = global_edge_index + edge_index_t

        # Global batch pointer
        global_batch_ptr = torch.repeat_interleave(
            torch.arange(0, num_subgraphs, device=device),
            k
        )

        # Encode with GNN
        subgraph_embeddings = self.gnn_encoder(
            x=global_x,
            edge_attr=global_edge_attr,
            edge_index=global_edge_index,
            batch=global_batch_ptr
        )

        return subgraph_embeddings

    def get_separation_stats(self) -> Dict:
        """
        Get statistics about the WL embedding layer.
        Useful for debugging and analysis.
        """
        return {
            'vocab_size': len(self.wl_vocab),
            'wl_dim': self.wl_dim,
            'gnn_dim': self.hidden_dim,
            'combined_dim': self.combined_dim,
            'frozen_params': sum(p.numel() for p in self.wl_embedding.parameters()),
            'trainable_params': sum(p.numel() for p in self.gnn_encoder.parameters() if p.requires_grad)
        }
