"""Epsilon-uniform connected subgraph sampler via random walk with rejection sampling."""

from typing import Tuple
import torch

def sample_batch(
    edge_index: torch.Tensor,
    ptr: torch.Tensor,
    m_per_graph: int,
    k: int,
    mode: str = "sample",
    seed: int = 42,
    epsilon: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample epsilon-uniform connected k-subgraphs from batched graphs.

    This sampler uses a random walk approach with rejection sampling to approximate
    uniform sampling. It is much faster than exact uniform sampling for large graphs,
    with the trade-off of approximate uniformity controlled by epsilon.

    Args:
        edge_index: [2, E] Edge connectivity tensor (int64)
        ptr: [B+1] Graph pointer tensor (int64)
        m_per_graph: Number of samples per graph
        k: Subgraph size
        mode: Sampling mode (currently only "sample" supported)
        seed: Random seed for reproducibility
        epsilon: Approximation parameter in (0, 1]. Smaller epsilon means closer to
                 uniform (but slower). Larger epsilon means faster (but less uniform).
                 Default: 0.1

    Returns:
        nodes_t: [total_samples, k] GLOBAL node IDs (for feature gathering)
        edge_index_t: [2, total_edges] LOCAL edge indices [0, k-1] per sample
        edge_ptr_t: [total_samples+1] Edge offsets per sample
        sample_ptr_t: [B+1] Sample offsets per graph
        edge_src_t: [total_edges] Original edge indices in input edge_index

    Epsilon-Uniformity Guarantee:
        Each connected k-subgraph is sampled with probability approximately
        proportional to 1/|C|, where |C| is the total number of connected
        k-subgraphs. The approximation quality is controlled by epsilon:
        - epsilon → 0: approaches truly uniform sampling
        - epsilon = 1: maximum speedup with reasonable uniformity

    Algorithm:
        Uses random walk expansion (BFS with random neighbor selection) combined
        with rejection sampling to correct for sampling bias. The acceptance
        probability is adjusted based on the generation probability of each sample.

    Note:
        Coordinate system for mode="sample":
        - nodes_t contains GLOBAL node IDs to gather features from batch
        - edge_index_t contains LOCAL indices [0, k-1] within each sample
        This matches the uniform_sampler and UGS sampler API.

        GPU Support:
        - Automatically handles GPU tensors (accepts cuda input)
        - Returns tensors on same device as input
        - Uses pinned memory for fast CPU↔GPU transfer
        - Sampling happens on CPU (optimized with OpenMP)
    """
    ...
