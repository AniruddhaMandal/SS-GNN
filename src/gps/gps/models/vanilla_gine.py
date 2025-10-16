import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool
from torch_geometric.nn.norm import BatchNorm


def make_mlp(in_dim, hidden_dim, out_dim, num_layers=2, activate_last=False):
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


class GINEClassifier(nn.Module):
    def __init__(self,
                 in_channels: int,
                 edge_dim: int,
                 hidden_dim: int,
                 num_classes: int = 10,
                 num_layers: int = 5,
                 mlp_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_classes = num_classes

        # Edge projection
        self.node_proj = nn.Linear(in_channels,hidden_dim)
        self.edge_proj = nn.Linear(edge_dim,hidden_dim)
        # First layer
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        mlp0 = make_mlp(hidden_dim, hidden_dim, hidden_dim, num_layers=mlp_layers)
        self.convs.append(GINEConv(mlp0, train_eps=True))
        self.batch_norms.append(BatchNorm(hidden_dim))

        # Remaining layers
        for _ in range(1, num_layers):
            mlp = make_mlp(hidden_dim, hidden_dim, hidden_dim, num_layers=mlp_layers)
            self.convs.append(GINEConv(mlp, train_eps=True))
            self.batch_norms.append(BatchNorm(hidden_dim))

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, edge_index, batch, edge_attr):
        h = self.node_proj(x)
        edge_attr = self.edge_proj(edge_attr)
        for conv, bn in zip(self.convs, self.batch_norms):
            h_res = h
            h = conv(h, edge_index, edge_attr)
            h = bn(h)
            h = F.relu(h)
            h = h + h_res  # residual
        
        g = global_mean_pool(h, batch)
        logits = self.classifier(g)

        probs = torch.sigmoid(logits)  # for multi-label AP
        preds = (probs > 0.5).float()
        one_hot = preds

        return logits, probs, preds, one_hot


# Quick smoke test
if __name__ == "__main__":
    from torch_geometric.data import Data, Batch

    num_graphs = 2
    n_nodes = 5
    in_channels = 16
    edge_dim = 8
    num_classes = 10

    data_list = []
    for _ in range(num_graphs):
        x = torch.randn(n_nodes, in_channels)
        edge_index = torch.tensor([[0, 1, 2, 3],
                                   [1, 2, 3, 4]])
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_attr = torch.randn(edge_index.size(1), edge_dim)
        data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))

    batch = Batch.from_data_list(data_list)
    model = GINEClassifier(in_channels, edge_dim, hidden_dim=64, num_classes=num_classes)
    out = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
    print("logits.shape:", out[0].shape)
