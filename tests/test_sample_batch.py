import torch
import torch_geometric as pyg
import ugs_sampler
from torch_geometric.loader import DataLoader

pepfunc_data = pyg.datasets.LRGBDataset("data", "Peptides-func")
loader = DataLoader(pepfunc_data,batch_size=4,shuffle = True)

for batch in loader:
    node_s, edge_index_s, edge_ptr_s, sample_ptr, edge_src_global_s = ugs_sampler.sample_batch(batch.edge_index,batch.ptr, 3, 10)
    print(f"Node:{node_s}")
    print(f"Edge Index: {edge_index_s}")
    print(f"Edge Ptr: {edge_ptr_s}")
    print(f"Sample Ptr:{sample_ptr}")
    print(f"Edge Source Global: {edge_src_global_s}")

    break