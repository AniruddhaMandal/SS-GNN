import torch 
import ugs_sampler

# ----- Build a tiny batched graph: two disjoint graphs -----
# Graph 0: nodes 0..3 ; edges: (0,1), (1,2), (2,3)
# Graph 1: nodes 4..7 ; edges: (4,5), (5,6), (6,7), (4,7)
ptr = torch.tensor([0, 4, 8], dtype=torch.long)  # [num_graphs+1]
edge_index = torch.tensor([
    [0, 1, 2, 4, 5, 6, 4],
    [1, 2, 3, 5, 6, 7, 7],
], dtype=torch.long)


m_per_graph = 2
k = 3


def test_sample_batch_global_rep():
    nodes_t, edge_index_t, edge_ptr_t, sample_ptr_t, edge_src_global_t = \
        ugs_sampler.sample_batch(edge_index, ptr, m_per_graph, k, mode="global")
    for i, e in enumerate(edge_index_t.t()):
        (u,v) = e
        (in_u, in_v) =edge_index.t()[edge_src_global_t[i]]
        assert ((u,v) == (in_u, in_v)) or ((u,v) == (in_v, in_u)), "[global]:In, Out nodes of edges of `edge_index_t` not in corresponding graph in `node_t`."

def get_subgraph_id_for_edge(i, edge_ptr_t):
    for j, (s,e) in enumerate(zip(edge_ptr_t[:-1],edge_ptr_t[1:])):
        if s<=i and i<e:
            return j

def test_sample_batch_sample():
    nodes_t, edge_index_t, edge_ptr_t, sample_ptr_t, edge_src_global_t = \
        ugs_sampler.sample_batch(edge_index, ptr, m_per_graph, k, mode="sample")
    
    for i,(u,v) in enumerate(edge_index_t.t()):
        g_id = get_subgraph_id_for_edge(i,edge_ptr_t)
        u_glob = nodes_t[g_id,u].item()
        v_glob = nodes_t[g_id,v].item()

        (u_org,v_org) = edge_index.t()[edge_src_global_t[i]]
        assert ((u_glob,v_glob) == (u_org, v_org)) or ((u_glob,v_glob) == (v_org, u_org)), "[sample]:In, Out nodes of edges of `edge_index_t` not in `edge_index` according to `edge_src_global_t`."
        