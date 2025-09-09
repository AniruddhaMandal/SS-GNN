import torch
import ugs_sampler
# small toy graph (triangle)
edge_index = torch.tensor([[0,1,2,3,4,5,5,6,7,8,9],
                           [1,2,3,4,5,0,6,7,8,9,0]], dtype=torch.long)

n = 3; k = 2

handle = ugs_sampler.create_preproc(edge_index, n, k)

nodes, edge_index_s, edge_ptr, graph_id = ugs_sampler.sample(handle, m_per_graph=10, k=k)

print(ugs_sampler.get_preproc_info(handle))
print("Sampled nodes:\n", nodes)
print("Sampled edge_index:\n", edge_index_s)
print("Edge ptr:", edge_ptr)
print("Graph id:", graph_id)

ugs_sampler.destroy_preproc(handle)
