"""
Sparse Clique Detection Dataset - Maximum Separability

Binary classification with extreme sparsity for high W-distance separation:
- Class 0: Very sparse graphs (tree-like) with NO k-cliques
- Class 1: Very sparse graphs + exactly ONE planted k-clique

Key Design:
- Base graphs are VERY sparse (p ≈ 0.01-0.02, almost tree-like)
- Class 1 has exactly one dense k-clique substructure
- This creates maximum distributional divergence for SS-GNN
"""

import random
from typing import Optional, Tuple, List, Callable

import torch
import networkx as nx
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from tqdm import tqdm


class SparseCliqueDetectionDataset(Dataset):
    """
    Extremely sparse clique detection for maximum SS-GNN separability.

    Class 0: Very sparse graphs (tree-like, no cliques)
    Class 1: Very sparse graphs + exactly 1 k-clique

    Parameters:
    - num_graphs: Total number of graphs (split evenly)
    - k: Clique size (default 4)
    - node_range: (min_n, max_n) for graph size
    - p_base: Edge probability for sparse base (default 0.015, very sparse)
    - seed: Random seed
    """

    def __init__(
        self,
        num_graphs: int = 2000,
        k: int = 4,
        node_range: Tuple[int, int] = (30, 50),
        p_base: float = 0.015,
        seed: Optional[int] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        device: str = "cpu",
        store_on_device: bool = False,
    ):
        super().__init__()

        self.num_graphs = int(num_graphs)
        self.k = int(k)
        self.min_n, self.max_n = int(node_range[0]), int(node_range[1])
        self.p_base = float(p_base)
        self.transform = transform
        self.pre_transform = pre_transform
        self.device = torch.device(device)
        self.store_on_device = bool(store_on_device)

        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        self.graphs: List[Data] = []
        self._generate_all_graphs()

    def _has_k_clique(self, G: nx.Graph) -> bool:
        """Check if graph contains a k-clique."""
        try:
            for clique in nx.find_cliques(G):
                if len(clique) >= self.k:
                    return True
            return False
        except:
            return False

    def _generate_class_0_graph(self) -> nx.Graph:
        """
        Generate very sparse graph WITHOUT any k-cliques.

        Strategy: Start with random tree, add few random edges, verify no clique.
        """
        n = random.randint(self.min_n, self.max_n)

        # Start with a tree (guaranteed no k-cliques for k >= 3)
        G = nx.random_labeled_tree(n)

        # Add a few random edges to make it slightly denser than a tree
        # but still very sparse
        num_extra_edges = int(n * self.p_base)
        added = 0
        max_attempts = n * 2

        for _ in range(max_attempts):
            if added >= num_extra_edges:
                break
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u != v and not G.has_edge(u, v):
                G.add_edge(u, v)
                # Verify we didn't accidentally create a k-clique
                if self._has_k_clique(G):
                    G.remove_edge(u, v)
                else:
                    added += 1

        return G

    def _generate_class_1_graph(self) -> nx.Graph:
        """
        Generate very sparse graph WITH exactly one k-clique.

        Strategy: Create sparse base graph, then plant one k-clique.
        """
        n = random.randint(self.min_n, self.max_n)

        # Start with a tree (very sparse base)
        G = nx.random_labeled_tree(n)

        # Add a few random edges (but keep it sparse)
        num_extra_edges = int(n * self.p_base * 0.5)  # Even sparser than class 0
        added = 0

        for _ in range(n * 2):
            if added >= num_extra_edges:
                break
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u != v and not G.has_edge(u, v):
                G.add_edge(u, v)
                added += 1

        # Plant exactly ONE k-clique
        if n >= self.k:
            clique_nodes = random.sample(list(G.nodes()), self.k)
            for i in range(len(clique_nodes)):
                for j in range(i + 1, len(clique_nodes)):
                    G.add_edge(clique_nodes[i], clique_nodes[j])

        return G

    def _nx_to_pyg(self, G: nx.Graph, label: int) -> Data:
        """Convert NetworkX graph to PyG Data."""
        data = from_networkx(G)
        data.y = torch.tensor([label], dtype=torch.long)
        data.num_nodes = G.number_of_nodes()
        data.num_edges = G.number_of_edges()

        if self.store_on_device:
            data = data.to(self.device)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        return data

    def _generate_all_graphs(self) -> None:
        """Generate all graphs for the dataset."""
        num_per_class = self.num_graphs // 2

        print(f"Generating {num_per_class} SPARSE graphs WITHOUT {self.k}-cliques...")
        for _ in tqdm(range(num_per_class), ncols=60, desc="[No Clique]"):
            G = self._generate_class_0_graph()
            data = self._nx_to_pyg(G, label=0)
            self.graphs.append(data)

        print(f"Generating {num_per_class} SPARSE graphs WITH one {self.k}-clique...")
        for _ in tqdm(range(num_per_class), ncols=60, desc="[Has Clique]"):
            G = self._generate_class_1_graph()
            data = self._nx_to_pyg(G, label=1)
            self.graphs.append(data)

        random.shuffle(self.graphs)

        # Statistics
        print(f"\n✓ Generated {len(self.graphs)} sparse graphs")
        print(f"  Class 0 (no {self.k}-clique): {sum(1 for g in self.graphs if g.y.item() == 0)}")
        print(f"  Class 1 (has {self.k}-clique): {sum(1 for g in self.graphs if g.y.item() == 1)}")

        avg_nodes = sum(g.num_nodes for g in self.graphs) / len(self.graphs)
        avg_edges = sum(g.num_edges for g in self.graphs) / len(self.graphs)
        avg_density = sum(2 * g.num_edges / (g.num_nodes * (g.num_nodes - 1))
                         for g in self.graphs if g.num_nodes > 1) / len(self.graphs)

        print(f"  Average nodes: {avg_nodes:.1f}")
        print(f"  Average edges: {avg_edges:.1f}")
        print(f"  Average density: {avg_density:.4f} (very sparse!)")

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Data:
        data = self.graphs[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data
