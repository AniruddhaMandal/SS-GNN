import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from typing import Tuple

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2):
        super().__init__()
        layers = []
        if num_layers == 1:
            layers.append(nn.Linear(in_dim, out_dim))
        else:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            for _ in range(num_layers-2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(hidden_dim, out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class SubgraphGINEncoder(nn.Module):
    """
    Encode sampled subgraphs with a small GIN and produce per-graph representations
    by averaging subgraph representations for each original graph.
    """
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int = 64,
                 num_gin_layers: int = 3,
                 mlp_layers: int = 2,
                 dropout: float = 0.0):
        super().__init__()
        self.in_channels = in_channels
        self.hidden = hidden_channels
        self.num_layers = num_gin_layers
        self.dropout = dropout

        # first linear to project input -> hidden
        self.node_proj = nn.Linear(in_channels, hidden_channels)

        # Build list of GINConv layers
        self.gin_convs = nn.ModuleList()
        for i in range(num_gin_layers):
            mlp = MLP(hidden_channels, hidden_channels, hidden_channels, num_layers=mlp_layers)
            conv = GINConv(mlp)
            self.gin_convs.append(conv)

        # final MLP to get subgraph representation (optional)
        self.final_lin = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels)
        )

    def forward(self,
                x_global: torch.Tensor,
                nodes_t: torch.LongTensor,
                edge_index_t: torch.LongTensor,
                edge_ptr_t: torch.LongTensor,
                graph_id_t: torch.LongTensor,
                k: int
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x_global : (N_total, F) tensor
            Node features for the batched graphs (global indexing).
        nodes_t : (B_total, k) int64 CPU/host or device tensor
            Each row lists k global node indices for that sampled subgraph (or -1 for missing).
        edge_index_t : (2, E_total) int64 tensor
            Concatenated local edge indices for all samples where values are in [0, k-1].
        edge_ptr_t : (B_total+1,) int64 tensor
            Edge pointer into edge_index_t for each sample.
        graph_id_t : (B_total,) int64 tensor
            For each sampled subgraph (row) the original graph id (0..G-1).
        k : int
            Number of nodes per sample (the fixed sample width).

        Returns
        -------
        per_graph_rep : (G, hidden) tensor
            Averaged subgraph representations for each original graph.
        subgraph_reps : (B_total, hidden) tensor
            Representation for each sampled subgraph (in sample order).
        """
        device = x_global.device
        # Move inputs to device
        nodes_t = nodes_t.to(device)
        edge_index_t = edge_index_t.to(device)
        edge_ptr_t = edge_ptr_t.to(device)
        graph_id_t = graph_id_t.to(device)

        B_total = nodes_t.size(0) # total number of graphs
        N_flat = B_total * k  # flattened nodes count (including placeholders -1)

        # --- 1) Build flattened node features for the batched subgraph nodes ---
        nodes_flat_idx = nodes_t.reshape(-1)             # (N_flat,)
        valid_mask = nodes_flat_idx != -1                # (N_flat,)
        # Prepare tensor for node features on device
        Fdim = x_global.size(1)
        node_feats = torch.zeros((N_flat, Fdim), device=device, dtype=x_global.dtype)

        if valid_mask.any():
            # gather only valid global indices
            valid_positions = valid_mask.nonzero(as_tuple=False).squeeze(1)
            valid_global_idx = nodes_flat_idx[valid_positions]    # indices into x_global
            node_feats[valid_positions] = x_global[valid_global_idx]


        # --- 3) Prepare 'batch' vector mapping each flattened node -> sample index (0..B_total-1) ---
        batch_nodes = torch.arange(N_flat, device=device) // k   # (N_flat,)

        # --- 4) Run GIN over the flattened node graph ---
        h = self.node_proj(node_feats)      # initial projection
        h = F.relu(h)
        for conv in self.gin_convs:
            if edge_index_t.size(1) == 0:
                # No edges: treat as isolated nodes (GINConv expects edges, so skip)
                h = F.relu(h)  # at least apply activation
            else:
                h = conv(h, edge_index_t)
                h = F.relu(h)
            if self.dropout > 0:
                h = F.dropout(h, p=self.dropout, training=self.training)

        # --- 5) Compute subgraph representations by mean over valid nodes in each sample ---
        # sum over nodes per sample:
        sub_sum = torch.zeros((B_total, h.size(1)), device=device, dtype=h.dtype)
        # counts per sample
        counts = torch.zeros((B_total,), device=device, dtype=h.dtype)
        # index_add works fine for 2D src
        sub_sum.index_add_(0, batch_nodes, h)
        counts.index_add_(0, batch_nodes, valid_mask.to(h.dtype))

        # avoid division by zero: for samples with zero valid nodes produce zero vector
        counts_clamped = counts.clone()
        counts_clamped[counts_clamped == 0] = 1.0
        subgraph_reps = sub_sum / counts_clamped.unsqueeze(1)   # (B_total, hidden)

        # Zero-out representations for truly empty samples (counts == 0)
        empty_mask = (counts == 0)
        if empty_mask.any():
            subgraph_reps[empty_mask] = 0.0

        # --- 6) Aggregate subgraph reps into per-graph (original graph) representations by averaging ---
        num_graphs = int(graph_id_t.max().item()) + 1 if graph_id_t.numel() > 0 else 0
        if num_graphs == 0:
            # nothing to aggregate
            per_graph_rep = torch.zeros((0, h.size(1)), device=device, dtype=h.dtype)
            return per_graph_rep, subgraph_reps

        per_sum = torch.zeros((num_graphs, h.size(1)), device=device, dtype=h.dtype)
        per_counts = torch.zeros((num_graphs,), device=device, dtype=h.dtype)
        per_sum.index_add_(0, graph_id_t, subgraph_reps)
        non_empty_mask = (counts > 0).to(h.dtype)   # counts computed earlier (nodes per sample)
        per_counts.index_add_(0, graph_id_t, non_empty_mask)

        per_counts_clamped = per_counts.clone()
        per_counts_clamped[per_counts_clamped == 0] = 1.0
        per_graph_rep = per_sum / per_counts_clamped.unsqueeze(1)
        per_graph_rep[per_counts == 0] = 0.0

        # optional final MLP
        per_graph_rep = self.final_lin(per_graph_rep)

        return per_graph_rep, subgraph_reps