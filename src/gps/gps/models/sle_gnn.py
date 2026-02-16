"""
Self-Loop Enhanced GNN (SLE-GNN)

Core Idea: At layer l, add l self-loops to each node before message passing.
This progressively emphasizes self-information as depth increases.

Reference: SLE_GNN notes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import (
    GCNConv, GINConv, GATConv, GATv2Conv, SAGEConv, SGConv, GCN2Conv, PNAConv,
    global_mean_pool, global_add_pool, global_max_pool
)
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.utils import add_self_loops, remove_self_loops
from gps import SubgraphFeaturesBatch
from typing import Optional


def _add_custom_self_loops(edge_index: Tensor, num_nodes: int, num_self_loops: int = 1) -> Tensor:
    """
    Add a specified number of self-loops to each node in the graph.

    Args:
        edge_index: [2, E] edge connectivity tensor
        num_nodes: Number of nodes in the graph
        num_self_loops: Number of self-loops to add per node

    Returns:
        edge_index: [2, E + num_nodes * num_self_loops] updated edge connectivity
    """
    if num_self_loops <= 0:
        return edge_index

    # Remove existing self-loops first to avoid duplicates
    edge_index, _ = remove_self_loops(edge_index)

    # Create self-loop edges: each node connects to itself
    self_loop_nodes = torch.arange(num_nodes, device=edge_index.device)

    # Repeat self-loops the specified number of times
    self_loop_edges = self_loop_nodes.repeat(num_self_loops)
    self_loop_index = torch.stack([self_loop_edges, self_loop_edges], dim=0)

    # Concatenate original edges with self-loops
    edge_index = torch.cat([edge_index, self_loop_index], dim=1)

    return edge_index


def make_mlp(in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2, activate_last: bool = False) -> nn.Sequential:
    """Create a multi-layer perceptron."""
    layers = []
    if num_layers == 1:
        layers += [nn.Linear(in_dim, out_dim)]
        if activate_last:
            layers += [nn.ReLU()]
    else:
        layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers += [nn.Linear(hidden_dim, out_dim)]
        if activate_last:
            layers += [nn.ReLU()]
    return nn.Sequential(*layers)


class SLEGNNLayer(nn.Module):
    """
    A single SLE-GNN layer that adds layer-specific self-loops before message passing.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_type: str = 'gcn',
        layer_idx: int = 1,
        mlp_layers: int = 2,
        heads: int = 1,
        dropout: float = 0.0,
        # GCNII parameters
        gcnii_alpha: float = 0.1,
        gcnii_theta: float = 0.5,
        # PNA parameters
        deg: torch.Tensor = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx  # 1-indexed: first layer adds 1 self-loop
        self.conv_type = conv_type.lower()
        self.dropout = dropout

        # Build the convolution layer
        if self.conv_type == 'gcn':
            self.conv = GCNConv(in_channels, out_channels, cached=False, normalize=True)
        elif self.conv_type == 'gin':
            mlp = make_mlp(in_channels, out_channels, out_channels, num_layers=mlp_layers)
            self.conv = GINConv(nn=mlp, train_eps=True)
        elif self.conv_type == 'gat':
            self.conv = GATConv(in_channels, out_channels // heads, heads=heads, concat=True, dropout=dropout)
        elif self.conv_type == 'gatv2':
            self.conv = GATv2Conv(in_channels, out_channels // heads, heads=heads, concat=True, dropout=dropout)
        elif self.conv_type == 'sage':
            self.conv = SAGEConv(in_channels, out_channels)
        elif self.conv_type == 'sgc':
            self.conv = SGConv(in_channels, out_channels, K=1)
        elif self.conv_type == 'gcnii':
            self.conv = GCN2Conv(out_channels, alpha=gcnii_alpha, theta=gcnii_theta,
                                 layer=layer_idx, shared_weights=True, cached=False, normalize=True)
        elif self.conv_type == 'pna':
            if deg is None:
                deg = torch.ones(128, dtype=torch.long)
            self.conv = PNAConv(in_channels, out_channels,
                                aggregators=['mean', 'min', 'max', 'std'],
                                scalers=['identity', 'amplification', 'attenuation'],
                                deg=deg)
        else:
            raise ValueError(f"Unknown conv_type: {conv_type}")

        self.bn = BatchNorm(out_channels)

    def forward(self, x: Tensor, edge_index: Tensor, num_nodes: int, x_0: Tensor = None) -> Tensor:
        """
        Forward pass with layer-specific self-loops.

        Args:
            x: [N, in_channels] node features
            edge_index: [2, E] edge connectivity
            num_nodes: number of nodes
            x_0: [N, in_channels] initial node features (required for GCNII)

        Returns:
            x: [N, out_channels] updated node features
        """
        # Add layer_idx self-loops to each node
        edge_index_with_loops = _add_custom_self_loops(edge_index, num_nodes, self.layer_idx)

        # Message passing
        if self.conv_type == 'gcnii':
            x = self.conv(x, x_0, edge_index_with_loops)
        else:
            x = self.conv(x, edge_index_with_loops)
        x = self.bn(x)
        x = F.relu(x)

        return x


class SLEGNNEncoder(nn.Module):
    """
    Self-Loop Enhanced GNN Encoder.

    At each layer l (1-indexed), adds l self-loops to each node before message passing.
    This progressively emphasizes self-information as network depth increases.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 3,
        conv_type: str = 'gcn',
        mlp_layers: int = 2,
        heads: int = 1,
        dropout: float = 0.1,
        residual: bool = True,
        jk_mode: Optional[str] = None,  # 'cat', 'max', 'lstm', or None
        # GCNII parameters
        gcnii_alpha: float = 0.1,
        gcnii_theta: float = 0.5,
        # PNA parameters
        deg: torch.Tensor = None,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.residual = residual

        # JKNet setup
        self.is_jknet = (conv_type.lower() == 'jknet')
        if self.is_jknet:
            self.jk_mode = jk_mode if jk_mode is not None else 'cat'
            conv_type = 'gcn'  # JKNet uses GCN as the base convolution

        # Input projection
        self.node_proj = nn.Linear(in_channels, hidden_dim)

        # PNA default degree histogram
        if conv_type == 'pna' and deg is None:
            deg = torch.ones(128, dtype=torch.long)

        # SLE-GNN layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_in = hidden_dim
            layer_out = hidden_dim
            self.layers.append(
                SLEGNNLayer(
                    in_channels=layer_in,
                    out_channels=layer_out,
                    conv_type=conv_type,
                    layer_idx=i + 1,  # 1-indexed
                    mlp_layers=mlp_layers,
                    heads=heads,
                    dropout=dropout,
                    gcnii_alpha=gcnii_alpha,
                    gcnii_theta=gcnii_theta,
                    deg=deg,
                )
            )

        # Jumping Knowledge aggregation
        if self.is_jknet:
            if self.jk_mode == 'cat':
                self.jk_linear = nn.Linear(hidden_dim * num_layers, out_dim)
                print("JK linear created.!\n\n")
            elif self.jk_mode == 'max':
                self.jk_linear = nn.Linear(hidden_dim, out_dim)
            elif self.jk_mode == 'lstm':
                self.jk_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
                self.jk_linear = nn.Linear(2 * hidden_dim, out_dim)
        else:
            self.out_proj = nn.Linear(hidden_dim, out_dim) if hidden_dim != out_dim else nn.Identity()

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """
        Forward pass through all SLE-GNN layers.

        Args:
            x: [N, in_channels] node features
            edge_index: [2, E] edge connectivity

        Returns:
            x: [N, out_dim] node embeddings
        """
        num_nodes = x.size(0)

        # Project to hidden dimension
        h = self.node_proj(x)

        # Store intermediate representations for JK
        layer_outputs = []

        # Store initial features for GCNII
        h_0 = h

        # Pass through SLE-GNN layers
        for i, layer in enumerate(self.layers):
            h_new = layer(h, edge_index, num_nodes, x_0=h_0)

            # Residual connection (skip for first layer if dimensions don't match)
            if self.residual:
                h_new = h_new + h

            # Dropout (skip on last layer)
            if i < self.num_layers - 1:
                h_new = F.dropout(h_new, p=self.dropout, training=self.training)

            h = h_new
            layer_outputs.append(h)

        # Jumping Knowledge aggregation
        if self.is_jknet:
            if self.jk_mode == 'cat':
                h = torch.cat(layer_outputs, dim=-1)
                h = self.jk_linear(h)
            elif self.jk_mode == 'max':
                h = torch.stack(layer_outputs, dim=0).max(dim=0)[0]
                h = self.jk_linear(h)
            elif self.jk_mode == 'lstm':
                h_stack = torch.stack(layer_outputs, dim=1)  # [N, L, H]
                h_lstm, _ = self.jk_lstm(h_stack)
                h = h_lstm[:, -1, :]  # Take last hidden state
                h = self.jk_linear(h)
        else:
            h = self.out_proj(h)

        return h


class SLEGNNClassifier(nn.Module):
    """
    SLE-GNN for graph-level classification.

    Combines SLE-GNN encoder with global pooling for graph classification tasks.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        out_dim: int = 2,
        num_layers: int = 3,
        conv_type: str = 'gcn',
        mlp_layers: int = 2,
        heads: int = 1,
        dropout: float = 0.1,
        pooling: str = 'mean',
        residual: bool = True,
        jk_mode: Optional[str] = None,
        edge_dim: Optional[int] = None,  # For compatibility, not used in base version
        gcnii_alpha: float = 0.1,
        gcnii_theta: float = 0.5,
        deg: torch.Tensor = None,
    ):
        super().__init__()

        self.encoder = SLEGNNEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            num_layers=num_layers,
            conv_type=conv_type,
            mlp_layers=mlp_layers,
            heads=heads,
            dropout=dropout,
            residual=residual,
            jk_mode=jk_mode,
            gcnii_alpha=gcnii_alpha,
            gcnii_theta=gcnii_theta,
            deg=deg,
        )

        # Global pooling
        if pooling == 'mean':
            self.pool = global_mean_pool
        elif pooling in ('add', 'sum'):
            self.pool = global_add_pool
        elif pooling == 'max':
            self.pool = global_max_pool
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        self.pooling_type = pooling

    def forward(self, batch: SubgraphFeaturesBatch) -> Tensor:
        """
        Forward pass for graph classification.

        Args:
            batch: SubgraphFeaturesBatch containing x, edge_index, batch

        Returns:
            graph_emb: [B, hidden_dim] graph embeddings
        """
        x = batch.x
        edge_index = batch.edge_index
        batch_idx = batch.batch

        # Encode nodes
        h = self.encoder(x, edge_index)

        # Global pooling
        graph_emb = self.pool(h, batch_idx)

        return graph_emb


class SLEGNNNodeClassifier(nn.Module):
    """
    SLE-GNN for node-level classification.

    Uses SLE-GNN encoder directly for node classification tasks.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 3,
        conv_type: str = 'gcn',
        mlp_layers: int = 2,
        heads: int = 1,
        dropout: float = 0.1,
        residual: bool = True,
        jk_mode: Optional[str] = None,
        gcnii_alpha: float = 0.1,
        gcnii_theta: float = 0.5,
        deg: torch.Tensor = None,
    ):
        super().__init__()

        self.encoder = SLEGNNEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            num_layers=num_layers,
            conv_type=conv_type,
            mlp_layers=mlp_layers,
            heads=heads,
            dropout=dropout,
            residual=residual,
            jk_mode=jk_mode,
            gcnii_alpha=gcnii_alpha,
            gcnii_theta=gcnii_theta,
            deg=deg,
        )

    def forward(self, batch: SubgraphFeaturesBatch) -> Tensor:
        """
        Forward pass for node classification.

        Args:
            batch: SubgraphFeaturesBatch containing x, edge_index

        Returns:
            node_emb: [N, hidden_dim] node embeddings
        """
        x = batch.x
        edge_index = batch.edge_index

        # Encode nodes
        h = self.encoder(x, edge_index)

        return h
