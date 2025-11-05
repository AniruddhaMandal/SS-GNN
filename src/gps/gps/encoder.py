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