import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GINEConv, GINConv, GCNConv, SAGEConv, GATv2Conv, global_mean_pool
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


class VanillaGNNClassifier(nn.Module):
    """
    A flexible graph-level classifier:
      - conv_type: 'gine' (uses edge_attr), 'gin', 'gcn', 'sage', 'gatv2'
      - For Peptides-func (multi-label), uses BCE-with-logits style outputs.
    """
    def __init__(self,
                 in_channels: int,
                 edge_dim: int,
                 hidden_dim: int,
                 num_classes: int = 10,
                 num_layers: int = 5,
                 mlp_layers: int = 2,
                 dropout: float = 0.1,
                 conv_type: str = 'gine'):
        super().__init__()
        conv_type = conv_type.lower()
        assert conv_type in {'gine', 'gin', 'gcn', 'sage', 'gatv2'}
        self.conv_type = conv_type
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_classes = num_classes

        # Project inputs to hidden_dim once so all layers are width-matched.
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

        # Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes)
        )

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

        g = global_mean_pool(h, batch)
        logits = self.classifier(g)                 # [B, num_classes]

        # Multi-label outputs (Peptides-func)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        one_hot = preds

        return logits, probs, preds, one_hot
