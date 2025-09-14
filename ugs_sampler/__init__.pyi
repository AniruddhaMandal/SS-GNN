def create_preproc(edge_index, num_nodes: int, k: int) -> int:
    """Builds Preprocessing 
    """
    ...
def destroy_preproc(handle: int) -> None: ...
def sample(handle: int, m_per_graph: int, k: int): ...
def has_graphlets(handle: int) -> bool: ...
def get_preproc_info(handle: int) -> dict: ...
def sample_batch(edge_index, prt, m_per_graph, k):
    """
    Batch-aware connected induced k-subgraph sampler.

    This function samples m_per_graph connected induced subgraphs of size k from each graph
    contained in a PyTorch-Geometric style batched graph. It calls the C++ preprocessing + sampler
    for every graph in the batch (using create_preproc + sample internally), then stitches the
    per-graph results into a single set of tensors that mirrors the output format of the underlying
    sample API.

    Parameters

    edge_index : torch.LongTensor, shape (2, M), device=cpu, dtype=torch.int64
    The batched edge index that concatenates the edges of all graphs (standard PyG format).
    Node indices in edge_index must be global indices in the range [0, total_num_nodes-1].
    Must be on CPU and int64.

    ptr : torch.LongTensor, shape (num_graphs + 1,), device=cpu, dtype=torch.int64
    The PyG Batch.ptr tensor describing node index ranges for each graph.
    Graph i has node indices ptr[i] .. ptr[i+1]-1.
    Must be on CPU and int64.

    m_per_graph : int
    Number of k-subgraphs to sample per graph.

    k : int
    Size (number of nodes) of each sampled connected induced subgraph.

    Returns

    nodes_t : torch.LongTensor, shape (G * m_per_graph, k), device=cpu, dtype=torch.int64, pinned
    Each row corresponds to a sampled subgraph (rows ordered by graph index then sample index).
    Entries are global node indices (relative to the input batched graph).
    If a sample failed / produced < k nodes the corresponding row will contain -1 for the
    missing positions.

    edge_index_t : torch.LongTensor, shape (2, E_total), device=cpu, dtype=torch.int64, pinned
    Concatenated edge lists for all sampled subgraphs. Each column is an (u, v) pair where
    u and v are local indices within the sample, i.e. values in 0 .. k-1.
    Use edge_ptr_t to locate the per-sample edge blocks.

    edge_ptr_t : torch.LongTensor, shape (B_total + 1,), device=cpu, dtype=torch.int64, pinned
    Cumulative pointer into edge_index_t of length B_total + 1, where
    B_total = num_graphs * m_per_graph. For sample b (0-based) its edges are
    edge_index_t[:, edge_ptr_t[b]:edge_ptr_t[b+1]].

    graph_id_t : torch.LongTensor, shape (B_total,), device=cpu, dtype=torch.int64, pinned
    For each sampled subgraph row, the graph_id_t[b] gives the original graph index
    (0 <= graph_id < num_graphs) from which this sample was drawn.

    Notes and important semantics

    The function treats the batched edge_index as a disjoint union of graphs and will not
    create subgraphs that mix nodes from different graphs: an edge belongs to graph i
    iff both endpoints are in ptr[i]:ptr[i+1].

    nodes_t rows contain global node indices (so you can directly index into the batched
    node feature tensor x using these indices after reshaping). The edge_index_t values,
    conversely, are local positions inside each nodes_t row: to read the actual endpoint
    node ids for an edge e in sample b you do:
    u_global = nodes_t[b, edge_index_t[0, e]]
    v_global = nodes_t[b, edge_index_t[1, e]]

    Failed samples: If the internal sampler could not build a size-k connected induced
    subgraph for a particular sample, the corresponding nodes_t row contains -1 values
    and the sample contributes no edges (edge block will be empty, i.e. edge_ptr_t entries equal).

    Device / dtype: All returned tensors are torch.int64 on CPU. They are created with
    pinned memory to facilitate fast host->device transfers if you later copy them to CUDA.

    Performance: This function builds preprocessing data per graph (CSR, orderings, buckets).
    That cost is O(n + m) per graph. Sampling uses the optimized C++ sampler and OpenMP where
    available. If you call sampling repeatedly on the same graphs, consider using the lower-level
    create_preproc once and calling sample(handle, ...) repeatedly to avoid recomputing the
    preprocess step (this helper currently creates and destroys per-graph preprocessing handles
    internally).

    Input validation: The function requires edge_index and ptr to be consistent. In particular,
    every index appearing in edge_index must satisfy 0 <= idx < ptr[-1]. If this is not true
    you will get assertions/errors. It is recommended to check:
    assert edge_index.min() >= 0 and edge_index.max() < ptr[-1]

    Thread-safety and robustness: The implementation creates temporary handles via the
    existing preprocessing API and destroys them when done. The sampler uses internal
    fallbacks for difficult neighborhoods, but depending on graph topology and k some samples
    might fail (see "Failed samples" above).

    Examples

    Basic usage with a PyG Batch:

    batch = next(iter(dataloader))      # batch.edge_index, batch.ptr
    nodes_t, edge_index_t, edge_ptr_t, graph_id_t = ugs_sampler_batch.sample_batch(
        batch.edge_index, batch.ptr, m_per_graph=10, k=5
    )

    # Example: convert the 0-th sample's edges to global node ids
    b = 0
    start, end = edge_ptr_t[b].item(), edge_ptr_t[b+1].item()
    for e in range(start, end):
        u_local = edge_index_t[0, e].item()
        v_local = edge_index_t[1, e].item()
        u_global = nodes_t[b, u_local].item()
        v_global = nodes_t[b, v_local].item()
        # now (u_global, v_global) are the global node ids in the batched graph

    Errors

    Raises RuntimeError if inputs are malformed (e.g. wrong dtype/device or inconsistent indexing).

    If the internal C++ sampler ultimately cannot produce any viable samples for all graphs
    it may raise an error; in practice the implementation includes fallbacks and will instead
    produce -1-filled rows for failed samples.

    """
    ...