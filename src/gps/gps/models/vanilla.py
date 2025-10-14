"""
Vanilla GIN (Graph Isomorphism Network) classifier implemented with PyTorch + PyTorch Geometric.

Outputs (in forward):
 - logits: raw class scores (shape [batch_size, num_classes])
 - probs: softmax probabilities over classes (shape [batch_size, num_classes])
 - preds: integer class predictions (shape [batch_size])
 - one_hot: one-hot encoding of preds (shape [batch_size, num_classes], float)

Usage notes:
 - Expects input as (x, edge_index, batch) like a standard PyG model.
 - For node-level tasks, remove global pooling and adapt classifier accordingly.
 - Requires `torch` and `torch_geometric` installed.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
from gps.registry import register_model


def make_mlp(in_dim, hidden_dim, out_dim, num_layers=2, activate_last=False):
    """Create a simple MLP used inside GINConv.

    Args:
        in_dim (int): input dimension
        hidden_dim (int): hidden dimension for intermediate layers
        out_dim (int): output dimension
        num_layers (int): total linear layers (>=1)
        activate_last (bool): whether to apply ReLU on last layer
    Returns:
        nn.Sequential MLP
    """
    assert num_layers >= 1
    layers = []
    if num_layers == 1:
        layers.append(nn.Linear(in_dim, out_dim))
        if activate_last:
            layers.append(nn.ReLU())
    else:
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, out_dim))
        if activate_last:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class GINClassifier(nn.Module):
    """A compact, configurable GIN classifier for graph-level classification.

    Parameters
    ----------
    in_channels : int
        Dimensionality of input node features.
    hidden_dim : int
        Hidden dimension for GIN layers and final classifier.
    num_classes : int
        Number of classes for classification.
    num_layers : int
        Number of GIN layers (>=1).
    mlp_layers : int
        Number of layers in the internal MLP used by each GINConv (>=1).
    dropout : float
        Dropout on the pooled graph representation before classifier.
    """

    def __init__(self,
                 in_channels: int,
                 hidden_dim: int,
                 num_classes: int,
                 num_layers: int = 3,
                 mlp_layers: int = 2,
                 dropout: float = 0.5):
        super().__init__()
        assert num_layers >= 1

        self.num_layers = num_layers
        self.dropout = dropout
        self.num_classes = num_classes

        # First GIN layer maps input features -> hidden
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # Layer 0
        mlp0 = make_mlp(in_channels, hidden_dim, hidden_dim, num_layers=mlp_layers)
        self.convs.append(GINConv(mlp0))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Remaining layers
        for _ in range(1, num_layers):
            mlp = make_mlp(hidden_dim, hidden_dim, hidden_dim, num_layers=mlp_layers)
            self.convs.append(GINConv(mlp))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Final classifier on pooled graph representation
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, edge_index, batch):
        """Forward pass.

        Args:
            x: node features tensor of shape [num_nodes, in_channels]
            edge_index: edge index tensor of shape [2, num_edges]
            batch: batch vector mapping nodes to example graph index [num_nodes]

        Returns:
            dict with keys: logits, probs, preds, one_hot
        """
        h = x
        for conv, bn in zip(self.convs, self.batch_norms):
            h = conv(h, edge_index)
            # batch norm expects shape [N, C]
            h = bn(h)
            h = F.relu(h)

        # Pool to get graph-level representation
        g = global_add_pool(h, batch)  # shape [batch_size, hidden_dim]

        logits = self.classifier(g)  # [batch_size, num_classes]

        probs = F.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        one_hot = F.one_hot(preds, num_classes=self.num_classes).to(dtype=logits.dtype)

        return logits,probs,preds.float(),one_hot


if __name__ == "__main__":
    # quick smoke test with random data
    try:
        from torch_geometric.data import Data, Batch
    except Exception:
        print("torch_geometric not found. Install torch_geometric to run the smoke test.")
    else:
        num_graphs = 4
        nodes_per_graph = 6
        in_channels = 8
        num_classes = 3

        # create a tiny batched dataset: simple chain per graph
        data_list = []
        for gi in range(num_graphs):
            n = nodes_per_graph
            x = torch.randn((n, in_channels))
            # create a chain
            edge_index = torch.stack([torch.arange(0, n - 1), torch.arange(1, n)])
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            data_list.append(Data(x=x, edge_index=edge_index))

        batch = Batch.from_data_list(data_list)

        model = GINClassifier(in_channels=in_channels, hidden_dim=32, num_classes=num_classes)
        out = model(batch.x, batch.edge_index, batch.batch)
        print("logits.shape:", out['logits'].shape)
        print("probs.sum(dim=1) (should be 1):", out['probs'].sum(dim=1))
        print("preds:", out['preds'])
        print("one_hot.shape:", out['one_hot'].shape)
