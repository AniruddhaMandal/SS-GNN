"""Truly uniform connected subgraph sampler via exhaustive enumeration."""

from typing import Tuple
import torch

def sample_batch(
    edge_index: torch.Tensor,
    ptr: torch.Tensor,
    m_per_graph: int,
    k: int,
    mode: str = "sample",
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample truly uniform connected k-subgraphs from batched graphs.

    Args:
        edge_index: [2, E] Edge connectivity tensor (int64)
        ptr: [B+1] Graph pointer tensor (int64)
        m_per_graph: Number of samples per graph
        k: Subgraph size
        mode: Sampling mode (currently only "sample" supported)
        seed: Random seed for reproducibility

    Returns:
        nodes_t: [total_samples, k] GLOBAL node IDs (for feature gathering)
        edge_index_t: [2, total_edges] LOCAL edge indices [0, k-1] per sample
        edge_ptr_t: [total_samples+1] Edge offsets per sample
        sample_ptr_t: [B+1] Sample offsets per graph
        edge_src_t: [total_edges] Original edge indices in input edge_index

    Note:
        Coordinate system for mode="sample":
        - nodes_t contains GLOBAL node IDs to gather features from batch
        - edge_index_t contains LOCAL indices [0, k-1] within each sample
        This matches the UGS sampler API.

        GPU Support:
        - Automatically handles GPU tensors (accepts cuda input)
        - Returns tensors on same device as input
        - Uses pinned memory for fast CPU↔GPU transfer
        - Enumeration happens on CPU (fast for small graphs)
    """
    ...
