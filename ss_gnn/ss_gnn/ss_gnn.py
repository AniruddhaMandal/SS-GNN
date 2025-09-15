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
    """SubgraphGINEncoder(in_channels, hidden_channels=64, num_gin_layers=3, mlp_layers=2, dropout=0.0)

    Encode sampled connected induced subgraphs with a small GIN and produce per-graph representations.

    Description
    -----------
    Given sampled k-node subgraphs (from the sampler), this module:
    1. Gathers node features for every sampled node (nodes are provided as global node ids).
    2. Runs a GIN over the flattened collection of sampled nodes + edges.
    3. Produces a subgraph representation by mean-pooling node embeddings inside each sample.
    4. Produces a graph-level representation by averaging its subgraph representations.

    Parameters
    ----------
    in_channels : int
        Number of input node features.
    hidden_channels : int, default=64
        Hidden dimension used inside GIN layers and final MLP.
    num_gin_layers : int, default=3
        Number of GIN message-passing layers.
    mlp_layers : int, default=2
        Number of layers inside each GIN's internal MLP.
    dropout : float, default=0.0
        Dropout applied after GIN layers (if > 0).

    Forward inputs
    --------------
    x_global : Tensor[N_total, F]
        Global node feature matrix for the batched graphs.
    nodes_t : LongTensor[B_total, k]
        Sample rows: each row contains k global node ids (or -1 for missing slots).
    edge_index_t : LongTensor[2, E_total]
        Edges for all samples. Can be local indices (0..k-1) or flattened indices
        (0..B_total*k-1) depending on the sampler mode.
    edge_ptr_t : LongTensor[B_total+1]
        Pointer into edge_index_t for per-sample edge blocks (may be unused if using flat edges).
    graph_id_t : LongTensor[B_total]
        For each sample row, the original graph index (0..G-1).
    k : int
        Number of nodes per sample (sampling width).

    Returns
    -------
    per_graph_rep : Tensor[G, hidden_channels]
        Per-graph representation obtained by averaging only *non-empty* subgraph representations.
    subgraph_reps : Tensor[B_total, hidden_channels]
        Representation for each sampled subgraph (empty samples -> zero vector).

    Notes
    -----
    - The encoder automatically moves sampler tensors to the device of `x_global`.
    - Placeholder slots in `nodes_t` are indicated by `-1`. Such nodes are ignored during gathering;
    samples with no valid nodes yield zero subgraph vectors and do not contribute to graph averages.
    - If using sampler edge_mode='flat', you can feed `edge_index_t` directly as flattened edges.
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


class SubgraphClassifier(nn.Module):
    """SubgraphClassifier(encoder, hidden_dim, num_classes=10, dropout=0.0)

    Classifier head that maps SubgraphGINEncoder graph representations to class logits.

    Parameters
    ----------
    encoder : nn.Module
        A SubgraphGINEncoder (or compatible) that returns (per_graph_rep, subgraph_reps).
    hidden_dim : int
        Dimensionality of the encoder's output per-graph representation (input to the head).
    num_classes : int, default=10
        Number of target classes.
    dropout : float, default=0.0
        Dropout probability applied in the head.

    Forward inputs
    --------------
    x_global : Tensor[N_total, F]
        Global node features for the batched graphs.
    nodes_t : LongTensor[B_total, k]
    edge_index_t : LongTensor[2, E_total]
    edge_ptr_t : LongTensor[B_total+1]
    graph_id_t : LongTensor[B_total]
    k : int
        Sampler outputs describing sampled subgraphs (see SubgraphGINEncoder docstring).

    Returns
    -------
    logits : Tensor[G, num_classes]
        Raw scores for each graph (use with CrossEntropyLoss).
    probs : Tensor[G, num_classes]
        Softmax probabilities.
    preds : LongTensor[G]
        Predicted class indices (argmax).
    one_hot : Tensor[G, num_classes]
        One-hot encoded predictions (float).

    Notes
    -----
    - `hidden_dim` must match the encoder's per-graph output size (or use an adapter layer).
    - For training, pass `logits` and a target LongTensor of shape (G,) with values in 0..num_classes-1
    to `torch.nn.CrossEntropyLoss`.
    """

    def __init__(self,
                 encoder: nn.Module,
                 hidden_dim: int,
                 num_classes: int = 10,
                 dropout: float = 0.0):
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        self.logit_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self,
                x_global: torch.Tensor,
                nodes_t: torch.LongTensor,
                edge_index_t: torch.LongTensor,
                edge_ptr_t: torch.LongTensor,
                graph_id_t: torch.LongTensor,
                k: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          logits: (num_graphs, num_classes)   -- raw scores (use with CrossEntropyLoss)
          probs:  (num_graphs, num_classes)   -- softmax probabilities
          preds:  (num_graphs,)                -- predicted class indices (long)
          one_hot: (num_graphs, num_classes)  -- one-hot encoded predictions (float)
        """
        # Get per-graph representation from encoder
        # encoder expected to return (per_graph_rep, subgraph_reps)
        per_graph_rep, _ = self.encoder(x_global, nodes_t, edge_index_t, edge_ptr_t, graph_id_t, k)

        logits = self.logit_head(per_graph_rep)            # (G, C)
        probs = F.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        one_hot = F.one_hot(preds, num_classes=self.num_classes).to(dtype=logits.dtype)

        return logits, probs, preds, one_hot

    def predict(self, *args, **kwargs):
        """Convenience: return predicted indices (cpu numpy)"""
        _, _, preds, _ = self.forward(*args, **kwargs)
        return preds.detach().cpu().numpy()
