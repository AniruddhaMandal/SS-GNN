import torch
import torch.nn as nn

class AtomBondEncoder:
    """Encode both atoms and bonds"""
    def __init__(self, 
                 atom_emb_dim: int = 64, 
                 bond_emb_dim: int = 32, 
                 atom_types: int = 28, 
                 bond_types: int = 4, 
                 requirs_grad: bool = False):
        self.atom_encoder = nn.Embedding(atom_types, atom_emb_dim)
        self.bond_encoder = nn.Embedding(bond_types, bond_emb_dim) if (bond_emb_dim is not None) else None
        for p in self.atom_encoder.parameters():
            p.requires_grad = requirs_grad
        for p in self.bond_encoder.parameters():
            p.requires_grad = requirs_grad
        
    def __call__(self, data):
        # Encode atoms
        x = data.x.view(-1)
        if (int(x.min())<0):
            raise ValueError(f"atom types must be >=0, got:{int(x.min())}")
        if (int(x.max())>self.atom_encoder.num_embeddings):
            raise ValueError(f"atom types too small: max {int(x.max())} >= {self.atom_encoder.num_embeddings}")
        data.x = self.atom_encoder(x.long())

        # Encode bonds
        if data.edge_attr is not None and self.bond_encoder is not None:
            ea = data.edge_attr
            # map to 1-base to 0-base edge type
            ea = ea-1
            if (int(ea.min()) < 0):
                raise ValueError(f"edge type must be >= 1, got {int(ea.min())}")
            if (int(ea.max()) >= self.bond_encoder.num_embeddings):
                raise ValueError(f"bond type too small, max {int(ea.max())}>={self.bond_encoder.num_embeddings}")
            data.edge_attr = self.bond_encoder(ea)
        return data

class FilterTarget:
    def __init__(self, target_idx):
        self.target_idx = target_idx

    def __call__(self, data):
        data.y = data.y[:,self.target_idx:self.target_idx+1]  # Keep dimension [N, 1]
        return data


class NormaliseTarget:
    """Normalizes targets based on training statistics

    This wrapper class normalizes targets using pre-computed or computed mean/std.
    Designed to be used after train/val/test split to avoid data leakage.
    """
    def __init__(self, dataset, mean=None, std=None, target_attr='y'):
        """
        Args:
            dataset: Dataset to wrap
            mean: Pre-computed mean (if None, compute from dataset)
            std: Pre-computed std (if None, compute from dataset)
            target_attr: Attribute name for targets (default: 'y')
        """
        self.dataset = dataset
        self.target_attr = target_attr

        if mean is None or std is None:
            # Compute stats from this dataset
            targets = []
            for i in range(len(dataset)):
                data = dataset[i]
                y = getattr(data, target_attr)
                targets.append(y)
            targets = torch.stack(targets)
            self.mean = targets.mean(dim=0)
            self.std = targets.std(dim=0)
        else:
            self.mean = mean
            self.std = std

    def __getitem__(self, idx):
        data = self.dataset[idx]
        y = getattr(data, self.target_attr)
        # Normalize
        normalized_y = (y - self.mean) / (self.std + 1e-8)
        setattr(data, self.target_attr, normalized_y)
        return data

    def __len__(self):
        return len(self.dataset)

    def get_stats(self):
        """Return normalization statistics for potential denormalization"""
        return self.mean, self.std