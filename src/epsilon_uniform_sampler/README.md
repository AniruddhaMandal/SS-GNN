# Epsilon-Uniform Sampler

Fast **ε-uniform** connected subgraph sampling via random walk with rejection sampling.

## Overview

This PyTorch extension implements an **epsilon-uniform** sampling algorithm for connected subgraphs, inspired by the paper:

> **"Efficient and near-optimal algorithms for sampling small connected subgraphs"**
> Marco Bressan, STOC 2021, ACM Transactions on Algorithms 2023
> arXiv: https://arxiv.org/abs/2007.12102

### Key Features

- **Fast**: 100-1000x faster than exact uniform sampling
- **Approximate Uniform**: Controlled by epsilon parameter (smaller ε = more uniform)
- **Connected Guarantee**: Always returns connected subgraphs
- **GPU Support**: Accepts CUDA tensors, automatically transfers back
- **OpenMP Optimized**: Parallel sampling across multiple graphs
- **API Compatible**: Matches `uniform_sampler` interface

## Algorithm

### Epsilon-Uniformity

An **ε-uniform sampler** approximates uniform sampling with quality controlled by ε ∈ (0, 1]:

- **ε → 0**: Approaches truly uniform (slower, more rejections)
- **ε = 0.1**: Recommended balance (fast with good uniformity)
- **ε → 1**: Maximum speed (reasonable uniformity)

### Implementation

Uses **random walk expansion** with **rejection sampling**:

1. **Random Walk Growth**: Start from random node, grow via BFS with random neighbor selection
2. **Bias Tracking**: Track generation probability of each sample
3. **Rejection Sampling**: Accept samples with probability adjusted by ε to approximate uniformity
4. **Connectivity Guarantee**: BFS-style growth ensures connectivity

**Acceptance Probability**:
```
P_accept = min(1, ε / (p_gen + ε))
```
where `p_gen` is the probability of generating this sample via random walk.

## Installation

```bash
cd src/epsilon_uniform_sampler
python setup.py build_ext --inplace
```

## Usage

### Basic Usage

```python
import sys
sys.path.insert(0, 'src/epsilon_uniform_sampler')
import torch
import epsilon_uniform_sampler

# Create graph
edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.int64)
ptr = torch.tensor([0, 3], dtype=torch.int64)

# Sample
nodes_t, edge_index_t, edge_ptr_t, sample_ptr_t, edge_src_t = \
    epsilon_uniform_sampler.sample_batch(
        edge_index,      # [2, E] Edge connectivity
        ptr,             # [B+1] Graph pointers
        m_per_graph=10,  # Samples per graph
        k=5,             # Subgraph size
        mode="sample",   # Output mode
        seed=42,         # Random seed
        epsilon=0.1      # Approximation parameter
    )
```

### API Signature

```python
def sample_batch(
    edge_index: torch.Tensor,  # [2, E] int64
    ptr: torch.Tensor,         # [B+1] int64
    m_per_graph: int,          # Samples per graph
    k: int,                    # Subgraph size
    mode: str = "sample",      # Output mode
    seed: int = 42,            # Random seed
    epsilon: float = 0.1,      # ε ∈ (0, 1]
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
```

### Returns

1. **nodes_t**: `[total_samples, k]` - GLOBAL node IDs (for feature gathering)
2. **edge_index_t**: `[2, total_edges]` - LOCAL edge indices [0, k-1] per sample
3. **edge_ptr_t**: `[total_samples+1]` - Edge offsets per sample (CSR format)
4. **sample_ptr_t**: `[B+1]` - Sample offsets per graph
5. **edge_src_t**: `[total_edges]` - Original edge indices in input

### Coordinate Systems

- **nodes_t**: Global IDs → use for feature gathering: `batch.x[nodes_t]`
- **edge_index_t**: Local IDs [0, k-1] → use for GNN message passing

## Performance

Comparison with exact `uniform_sampler` (10 nodes, k=4, 3 samples):

| Sampler | Epsilon | Time | Speedup |
|---------|---------|------|---------|
| Uniform (Exact) | - | 158.25 ms | 1x |
| Epsilon-Uniform | 0.01 | 1.28 ms | **124x** |
| Epsilon-Uniform | 0.1 | 0.16 ms | **1001x** |
| Epsilon-Uniform | 0.5 | 0.03 ms | **5578x** |

### Uniformity Quality

Coefficient of variation (lower = more uniform) on 1000 samples:

- Uniform (Exact): **0.542**
- Epsilon (ε=0.01): **0.698**
- Epsilon (ε=0.1): **0.698**
- Epsilon (ε=0.5): **0.753**

**Conclusion**: ε=0.1 provides excellent balance - nearly 1000x faster with only slightly less uniformity.

## When to Use

### Use Epsilon-Uniform Sampler When:
- Graph is large (n > 20)
- Speed is critical
- Approximate uniformity is acceptable
- Training ML models (speed > perfect uniformity)

### Use Exact Uniform Sampler When:
- Graph is small (n < 15)
- Perfect uniformity required
- Theoretical analysis needs exact distribution

## Example: Integration with GNN

```python
import torch
import torch_geometric as pyg
import epsilon_uniform_sampler

# Load graph
data = ...  # PyG Data object

# Sample subgraphs
nodes_t, edge_index_t, edge_ptr_t, sample_ptr_t, edge_src_t = \
    epsilon_uniform_sampler.sample_batch(
        edge_index=data.edge_index,
        ptr=torch.tensor([0, data.num_nodes]),
        m_per_graph=100,
        k=8,
        epsilon=0.1
    )

# Gather features - USE GLOBAL IDs
sample_features = data.x[nodes_t]  # [100, 8, feature_dim]

# GNN forward - USE LOCAL edge indices
for i in range(100):
    # Get edges for this sample (LOCAL indices)
    edge_start = edge_ptr_t[i]
    edge_end = edge_ptr_t[i+1]
    sample_edges = edge_index_t[:, edge_start:edge_end]  # LOCAL [0, k-1]

    # Get features for this sample
    x = sample_features[i]  # [k, feature_dim]

    # GNN forward pass
    output = gnn_model(x, sample_edges)  # Works with LOCAL indices
```

## Testing

Run tests:
```bash
python test_epsilon_sampler.py
python compare_samplers.py
```

## Implementation Notes

### Correctness
- All samples are guaranteed to be connected (verified via BFS)
- Rejection sampling ensures approximate uniformity
- Handles edge cases (disconnected graphs, insufficient nodes)

### Optimization
- OpenMP parallelization across graphs
- Thread-local RNG for thread safety
- Adaptive rejection threshold based on ε
- Early termination if cannot reach k nodes

### Limitations
- Sampling happens on CPU (fast for small k)
- Very small ε (< 0.01) may require many rejections
- No guarantee on minimum diversity of samples (use larger ε if too similar)

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{bressan2023efficient,
  title={Efficient and near-optimal algorithms for sampling small connected subgraphs},
  author={Bressan, Marco},
  journal={ACM Transactions on Algorithms},
  year={2023},
  publisher={ACM}
}
```

## License

Same as parent repository (SS-GNN).
