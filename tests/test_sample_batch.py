import torch
import torch_geometric as pyg
import ugs_sampler
from torch_geometric.loader import DataLoader

pepfunc_data = pyg.datasets.LRGBDataset("data", "Peptides-func")
loader = DataLoader(pepfunc_data,batch_size=64,shuffle = True)

for batch in loader:
    node_s, edge_index_s, edge_ptr_s, graph_id = ugs_sampler.sample_batch(batch.edge_index,batch.ptr, 300, 20)
    print(f"Node:{node_s}")
    print(f"Edge Index: {edge_index_s}")
    print(f"Edge Ptr: {edge_index_s}")
    print(f"Graph Id:{graph_id}")
    break