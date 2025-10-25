import torch

def create_preproc(edge_index, num_nodes: int, k: int) -> int:
    """Builds Preprocessing 
    """
    ...
def destroy_preproc(handle: int) -> None: ...
def sample(handle: int, m_per_graph: int, k: int,edge_mode:str = "local", base_offset: int = 0): ...
def has_graphlets(handle: int) -> bool: ...
def get_preproc_info(handle: int) -> dict: ...
def sample_batch(edge_index: torch.Tensor,
                 ptr: torch.Tensor,
                 m_per_graph: int,
                 k: int,
                 mode: str = "sample"):
    """
    Sample a fixed number of subgraphs from each graph in a batched graph dataset.

    Args:
        edge_index (torch.Tensor): 
            Global batched edge index of shape [2, E_total], with edges concatenated
            across all graphs in the batch.
        ptr (torch.Tensor): 
            Pointer tensor of shape [num_graphs + 1], where ptr[i] and ptr[i+1]
            mark the node index range [start, end) of the i-th graph in the batch.
        m_per_graph (int): 
            Number of subgraphs to sample per input graph.
        k (int): 
            Number of nodes per subgraph.
        mode (str): 
            Node indexing mode for the returned edge_index. One of:
                - "sample": indices are local to each subgraph (0..k-1)
                - "graph":  indices are local to the parent graph
                - "global": indices are global within the batch
    Returns:
        nodes_t (torch.Tensor):
            Tensor of shape [B_total, k], where B_total = num_graphs * m_per_graph. 
            Each row lists the global node IDs of the sampled subgraph (padded with -1).

        edge_index_t (torch.Tensor):
            Tensor of shape [2, E_total_sampled], concatenating edges from all sampled subgraphs. 
            Node indices are represented according to the chosen `mode`.

        edge_ptr_t (torch.Tensor):
            Tensor of shape [B_total + 1], CSR-style pointer array indicating the start
            and end positions in `edge_index_t` for each sampled subgraph.

        sample_ptr_t (torch.Tensor):
            Tensor of shape [num_graphs + 1], pointer array mapping subgraphs back to
            their parent graphs.

        edge_src_global_t (torch.Tensor):
            Tensor of shape [E_total_sampled], giving the original edge column indices
            in the input `edge_index` corresponding to each sampled edge. This can be used
            to gather edge attributes for the subgraphs.
    
    Notes:
        - Edge attributes can be aligned via:
              sub_edge_attr = edge_attr[edge_src_global_t]
    """
    ...