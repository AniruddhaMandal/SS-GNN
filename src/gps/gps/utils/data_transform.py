import torch
import torch.nn.functional as F
from torch_geometric.utils import degree
from torch_geometric.transforms import Compose, ToUndirected
from torch_geometric.datasets import TUDataset

class ClipOneHotDegree:
    def __init__(self, max_degree, cat=False):
        self.max_degree = max_degree
        self.cat = cat
    def __call__(self, data):
        deg = degree(data.edge_index[0], num_nodes=data.num_nodes).to(torch.long)
        deg = deg.clamp_max(self.max_degree)  # overflow → last bin
        x_oh = F.one_hot(deg, num_classes=self.max_degree + 1).float()
        data.x = x_oh if not self.cat else torch.cat([data.x, x_oh], dim=-1)
        return data

if __name__ == "__main__":
    transforms = Compose([ToUndirected(), ClipOneHotDegree(max_degree=512, cat=False)])  # pick a roomy cap
    dataset = TUDataset(root="data/TUDataset", name="COLLAB", transform=transforms)
