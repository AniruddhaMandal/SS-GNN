import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn import (
    GINEConv, GINConv, GCNConv, SAGEConv, GATv2Conv, global_mean_pool, global_add_pool, global_max_pool, global_sort_pool
)
from torch_geometric.nn.norm import BatchNorm


def make_mlp(in_dim, hidden_dim, out_dim, num_layers=2, activate_last=False):
    layers = []
    if num_layers == 1:
        layers += [nn.Linear(in_dim, out_dim)]
        if activate_last: layers += [nn.ReLU()]
    else:
        layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers += [nn.Linear(hidden_dim, out_dim)]
        if activate_last: layers += [nn.ReLU()]
    return nn.Sequential(*layers)


class SubgraphGNNEncoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 edge_dim: int, 
                 hidden_dim: int,
                 num_layers: int = 5,
                 mlp_layers: int = 4,
                 dropout: float = 0.1,
                 conv_type: str = "gine",
                 pooling: str = "mean"):
        super().__init__()
        conv_type = conv_type.lower()
        assert conv_type in {'gine', 'gin', 'gcn', 'sage', 'gatv2'}
        self.conv_type = conv_type
        self.num_layers = num_layers
        self.dropout = dropout
        self.out_channels = out_channels
        if pooling == 'mean':
            self.pooling = global_mean_pool
        elif pooling == 'add':
            self.pooling = global_add_pool
        elif pooling == 'max':
            self.pooling = global_max_pool
        else:
            raise ValueError(f"unknown value of subgraph pooling: {pooling}")

        self.node_proj = nn.Linear(in_channels, hidden_dim)

        # If using edge features (only GINE uses them directly)
        self.use_edges = (conv_type == 'gine')
        if self.use_edges:
            # Project raw edge_attr -> hidden_dim once (works for all layers)
            self.edge_proj = nn.Linear(edge_dim, hidden_dim)

        # Build layers
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(self._make_conv(hidden_dim, hidden_dim, mlp_layers))
            self.bns.append(BatchNorm(hidden_dim))

    def _make_conv(self, in_dim, out_dim, mlp_layers):
        if self.conv_type == 'gine':
            mlp = make_mlp(in_dim, in_dim, out_dim, num_layers=mlp_layers)
            return GINEConv(nn=mlp, train_eps=True)  # we project edge_attr ourselves
        if self.conv_type == 'gin':
            mlp = make_mlp(in_dim, in_dim, out_dim, num_layers=mlp_layers)
            return GINConv(nn=mlp, train_eps=True)
        if self.conv_type == 'gcn':
            return GCNConv(in_dim, out_dim, cached=False, normalize=True)
        if self.conv_type == 'sage':
            return SAGEConv(in_dim, out_dim)
        if self.conv_type == 'gatv2':
            return GATv2Conv(in_dim, out_dim, heads=1, concat=True)  # keeps dim = out_dim
        raise ValueError("Unknown conv_type")

    # ---------- FORWARD ----------
    def forward(self, x, edge_index, batch, edge_attr=None):
        """
        x: [N, in_channels]
        edge_index: [2, E]
        batch: [N]
        edge_attr: [E, edge_dim]  (required if conv_type='gine')
        """
        h = self.node_proj(x)
        if self.use_edges:
            if edge_attr is None:
                raise ValueError("edge_attr is required for conv_type='gine'.")
            e = self.edge_proj(edge_attr)

        for conv, bn in zip(self.convs, self.bns):
            h_res = h
            if self.use_edges:
                h = conv(h, edge_index, e)          # GINE expects edge features
            else:
                h = conv(h, edge_index)             # GIN/GCN/SAGE/GATv2 ignore edge_attr
            h = bn(h)
            h = F.relu(h)
            h = h + h_res                           # residual (dims match)

        g = self.pooling(h, batch)

        return g

'''
class EncodingAggregatorTransformer(nn.Module):
    """
    Aggregates per-subgraph encodings -> per-graph encodings via a Transformer.
    - Inserts a learned [CLS] per graph and returns the [CLS] output as the graph embedding.
    - Handles variable #subgraphs per graph with key_padding_mask.
    - Optional learned/sinusoidal positional encodings over the (sample) order.
    """
    def __init__(self,
                 d_model: int = 256,
                 n_heads: int = 8,
                 num_layers: int = 2,
                 dim_feedforward: int = 4 * 256,
                 dropout: float = 0.1,
                 positional_encoding: str = "learned",   # "learned" | "sinusoidal" | "none"
                 max_samples: int = 1024,                # max subgraphs per graph (for PE table)
                 use_cls: bool = True,
                 out_norm: bool = True):
        super().__init__()
        self.d_model = d_model
        self.use_cls = use_cls
        self.positional_encoding = positional_encoding

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,  # Transformer expects [S, B, E]
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        if positional_encoding == "learned":
            self.pos_emb = nn.Embedding(max_samples + (1 if use_cls else 0), d_model)
        elif positional_encoding == "sinusoidal":
            self.register_buffer("pos_sin", self._build_sinusoidal_pe(max_samples + (1 if use_cls else 0), d_model),
                                 persistent=False)
        else:
            self.pos_emb = None

        if use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.out_norm = nn.LayerNorm(d_model) if out_norm else nn.Identity()

    def forward(self,
                sample_emb: torch.Tensor,  # [B_total, D]
                graph_id_t: torch.Tensor   # [B_total]  ints 0..G-1
                ) -> torch.Tensor:
        """
        Returns:
            graph_emb: [G, D]
        """
        device = sample_emb.device
        D = sample_emb.size(-1)
        assert D == self.d_model, f"d_model mismatch: got {D}, expected {self.d_model}"

        # --- 1) Split sample_emb into a list of tensors, one per graph (lengths can vary)
        # Assumption: rows are ordered by graph id then sample index (your sampler does this).
        num_graphs = int(graph_id_t.max().item()) + 1 if graph_id_t.numel() else 0

        # Efficient slicing when samples are already grouped: find boundaries via bincount.
        counts = torch.bincount(graph_id_t, minlength=num_graphs)
        # Make cumulative starts [0, c0, c0+c1, ...]
        starts = torch.zeros_like(counts)
        starts[1:] = torch.cumsum(counts, dim=0)[:-1]
        # Build list: samples_g[g] = sample_emb[starts[g]:starts[g]+counts[g]]
        samples_g = [sample_emb[starts[g]: starts[g] + counts[g]] for g in range(num_graphs)]

        # Edge case: empty batch
        if num_graphs == 0:
            return sample_emb.new_zeros(0, D)

        # --- 2) (Optional) prepend a [CLS] token to each sequence
        if self.use_cls:
            cls = self.cls_token.expand(num_graphs, -1, -1)  # [G, 1, D]
            seqs = [torch.cat([cls[g], samples_g[g]], dim=0) for g in range(num_graphs)]  # each [S_g+1, D]
        else:
            seqs = samples_g  # each [S_g, D]

        # --- 3) Pad to [G, S_max, D] and build key_padding_mask [G, S_max]
        # pad_sequence expects list of [S_i, D] -> [S_max, G, D] when batch_first=False
        # We'll first pad as batch_first=True for clarity, then permute.
        padded = pad_sequence(seqs, batch_first=True)  # [G, S_max, D]
        # Mask: True where we padded
        lengths = torch.tensor([s.size(0) for s in seqs], device=device)  # [G]
        S_max = int(lengths.max().item())
        arange_S = torch.arange(S_max, device=device).unsqueeze(0)        # [1, S_max]
        key_padding_mask = arange_S >= lengths.unsqueeze(1)               # [G, S_max], True = pad

        # --- 4) Add positional encodings
        if self.positional_encoding == "learned":
            # positions 0..S_max-1 ; if using CLS, position 0 is CLS
            pos_ids = arange_S  # [1, S_max]
            pos = self.pos_emb(pos_ids).expand(num_graphs, -1, -1)  # [G, S_max, D]
            padded = padded + pos
        elif self.positional_encoding == "sinusoidal":
            pos = self.pos_sin[:S_max].unsqueeze(0).expand(num_graphs, -1, -1)  # [G, S_max, D]
            padded = padded + pos
        # else: no positional encoding

        # --- 5) Transformer expects [S, B, E]
        src = padded.permute(1, 0, 2).contiguous()            # [S_max, G, D]
        # key_padding_mask is [B, S] where True means "ignore"
        out = self.encoder(src, src_key_padding_mask=key_padding_mask)  # [S_max, G, D]

        # --- 6) Pool to per-graph
        if self.use_cls:
            graph_emb = out[0]                                # [G, D] -> first token ([CLS])
        else:
            # mean over valid tokens only
            out_bt = out.permute(1, 0, 2)                     # [G, S_max, D]
            valid = (~key_padding_mask).float().unsqueeze(-1) # [G, S_max, 1]
            summed = (out_bt * valid).sum(dim=1)              # [G, D]
            counts = valid.sum(dim=1).clamp(min=1.0)          # [G, 1]
            graph_emb = summed / counts

        return self.out_norm(graph_emb)                       # [G, D]

    @staticmethod
    def _build_sinusoidal_pe(max_len: int, d_model: int) -> torch.Tensor:
        """Classic Transformer sinusoidal PE table: [max_len, d_model]."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
'''

class SubgraphSamplingGNNClassifier(nn.Module):
    """
    Graph-level classification model with subgraph sampling and configurable GNN backbone.

    This classifier encodes subgraphs using a flexible GNN encoder, aggregates
    their representations at the graph level, and predicts class logits. 
    It supports multiple GNN convolution types and pooling strategies.

    Args:
        in_channels (int): Number of input node feature dimensions.
        edge_dim (int): Number of edge feature dimensions (used if conv_type='gine').
        hidden_dim (int): Hidden embedding dimension for node and graph representations.
        num_classes (int, optional): Number of output classes. Defaults to 10.
        num_layers (int, optional): Number of GNN layers in the encoder. Defaults to 5.
        mlp_layers (int, optional): Number of MLP layers within each GNN block. Defaults to 2.
        dropout (float, optional): Dropout probability applied in the classifier. Defaults to 0.1.
        conv_type (str, optional): Type of GNN convolution to use. One of:
            {'gine', 'gin', 'gcn', 'sage', 'gatv2'}. Defaults to 'gine'.
        pooling (str, optional): Type of graph-level pooling. Defaults to 'mean'.

    Inputs:
        x_global (Tensor): Node features of shape [N, in_channels].
        edge_attr (Tensor): Edge features of shape [E, edge_dim] (required for GINE).
        nodes_t (Tensor): Flattened node indices per sampled subgraph.
        edge_index_t (Tensor): Edge indices for sampled subgraphs [2, E_sub].
        edge_ptr_t (Tensor): Pointer array delimiting edges per subgraph.
        sample_ptr_t (Tensor): Pointer array delimiting subgraphs per graph.
        edge_src_global_t (Tensor): Source node indices in global node space.

    Returns:
        Tensor: Logits of shape [G, num_classes], where G is the number of graphs in the batch.

    Note:
        - Supports both single-label and multi-label classification (e.g., Peptides-func).
        - When `conv_type='gine'`, `edge_attr` must be provided.
    """
    def __init__(self,
                 in_channels: int,
                 edge_dim: int,
                 hidden_dim: int,
                 num_classes: int = 10,
                 num_layers: int = 5,
                 mlp_layers: int = 2,
                 dropout: float = 0.1,
                 conv_type: str = 'gine',
                 graph_level_pooling: str = 'mean',
                 subgraph_level_pooling: str = 'mean'):
                 
        super().__init__()

        # set encoder
        self.encoder = SubgraphGNNEncoder(in_channels=in_channels,
                                          out_channels=hidden_dim,
                                          hidden_dim=hidden_dim,
                                          edge_dim=edge_dim,
                                          num_layers=num_layers,
                                          conv_type=conv_type,
                                          mlp_layers=mlp_layers,
                                          dropout=dropout,
                                          pooling=subgraph_level_pooling)

        # set aggregator
        if graph_level_pooling == "mean":
            self.aggregator = global_mean_pool
        elif graph_level_pooling == "add":
            self.aggregator = global_add_pool
        elif graph_level_pooling == "max":
            self.aggregator = global_max_pool
        else:
            raise ValueError(f"unknown pooling type:{graph_level_pooling}")

        # Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    # ---------- FORWARD ----------
    def forward(self, 
                x_global, 
                edge_attr,
                nodes_t, 
                edge_index_t, 
                edge_ptr_t,
                sample_ptr_t,
                edge_src_global_t
                ):
        """
        x: [N, in_channels]
        edge_index: [2, E]
        edge_attr: [E, edge_dim]  (required if conv_type='gine')
        """
        device = x_global.device
        batch_size = sample_ptr_t.size(0)-1
        sample_emb = self.encode_subgraphs(x=x_global,
                                           edge_attr=edge_attr,
                                            nodes_t=nodes_t,
                                            edge_index_t=edge_index_t,
                                            edge_ptr_t=edge_ptr_t,
                                            edge_src_global_t=edge_src_global_t)  # [B_total, H_dim]

        global_graph_ptr = torch.repeat_interleave(torch.arange(0,batch_size, device=device),sample_ptr_t[1:]-sample_ptr_t[:-1])
        graph_emb = self.aggregator(sample_emb, global_graph_ptr)  # [num_graph, H]

        logits = self.classifier(graph_emb)  # [G, num_classes]
        return logits

    def encode_subgraphs(
        self,
        x,                 # (N, IN_dim)
        edge_attr,         # (E_global, E_dim)
        nodes_t,           # (B_total, k)  global node ids; -1 padded
        edge_index_t,      # (2, E_total)  in local/sample mode(in (0,1,..k-1)), concatenated
        edge_ptr_t,        # (B_total+1,)
        edge_src_global_t  # (E_total,) index of edges from `edge_index_t` in batch.edge_index
    ):
        """
        Returns:
            sample_emb: (B_total, H_dim)
        """
        device = x.device
        num_subgraphs, k = nodes_t.shape
        
        # gather global node attrib 
        stacked_nodes = nodes_t.flatten()
        global_x = x[stacked_nodes]

        # gather global edge atrribute
        global_edge_attr = edge_attr[edge_src_global_t]

        # convert edge index to batch(subgraph) level
        global_edge_index = torch.repeat_interleave(torch.arange(0,num_subgraphs,device=device),edge_ptr_t[1:]-edge_ptr_t[:-1])*k
        global_edge_index = global_edge_index + edge_index_t

        # global batch pointer 
        global_batch_ptr = torch.repeat_interleave(torch.arange(0,num_subgraphs,device=device),k)
        
        # send to device
        x = self.encoder(x=global_x,edge_attr=global_edge_attr,edge_index=global_edge_index,batch=global_batch_ptr)

        return x