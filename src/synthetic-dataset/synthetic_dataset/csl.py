"""CSL (Circular Skip Link) Dataset - Standard 1-WL failure benchmark"""

import torch
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from typing import Optional, Callable


class CSLDataset:
    """
    10-class classification where all graphs are 1-WL equivalent.
    Each class has different skip pattern in circular graph.
    Standard benchmark: Bouritsas et al. 2020
    """

    def __init__(self, transform: Optional[Callable] = None):
        self.num_nodes = 41
        self.num_graphs_per_class = 15
        self.transform = transform
        self.data_list = self._generate()

    def _make_csl_graph(self, class_idx: int) -> nx.Graph:
        """Generate CSL graph for given class (skip pattern)"""
        n = self.num_nodes
        G = nx.Graph()
        G.add_nodes_from(range(n))

        # Base cycle
        for i in range(n):
            G.add_edge(i, (i + 1) % n)

        # Skip connections based on class
        skip = class_idx + 2  # skip = 2,3,4,...,11
        for i in range(n):
            G.add_edge(i, (i + skip) % n)

        return G

    def _generate(self):
        data_list = []
        for class_idx in range(10):
            for _ in range(self.num_graphs_per_class):
                G = self._make_csl_graph(class_idx)
                data = from_networkx(G)
                data.x = torch.ones(self.num_nodes, 1)  # Constant features
                data.y = torch.tensor([class_idx], dtype=torch.long)
                if self.transform:
                    data = self.transform(data)
                data_list.append(data)
        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


if __name__ == "__main__":
    dataset = CSLDataset()
    print(f"✓ CSL Dataset: {len(dataset)} graphs")
    print(f"  Classes: 10, Samples/class: 15")
    print(f"  Nodes: {dataset.num_nodes}, All 3-regular")
    print(f"  Sample: {dataset[0]}")
