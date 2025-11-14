"""Random-Walk-with-Restart connected subgraph sampler (pyi stub)"""

from typing import Tuple
import torch

def sample_batch(
    edge_index: torch.Tensor,
    ptr: torch.Tensor,
    m_per_graph: int,
    k: int,
    mode: str = "sample",
    seed: int = 42,
    p_restart: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample connected induced k-subgraphs using Random-Walk-with-Restart (RWR).

    Args:
        edge_index: [2, E] int64 tensor (global node IDs)
        ptr: [B+1] int64 tensor, graph node pointers (CSR-like)
        m_per_graph: samples per graph
        k: target subgraph size
        mode: "sample" (local indexing inside sample) or "global"
        seed: RNG seed
        p_restart: restart probability in [0,1]

    Returns:
        nodes_t: [total_samples, k] GLOBAL node IDs
        edge_index_t: [2, total_edges] local/global edge indices depending on mode
        edge_ptr_t: [total_samples+1] edge offsets per sample
        sample_ptr_t: [B+1] sample offsets per graph
        edge_src_t: [total_edges] original edge indices (or -1 if not tracked)
    """
    ...
