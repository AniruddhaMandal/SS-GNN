# Uniform Sampler

**Truly uniform connected subgraph sampling via exhaustive enumeration.**

## Overview

This module provides **provably uniform** sampling of connected k-subgraphs from graphs, unlike the UGS sampler which uses an approximation that introduces bias.

### Performance

**On QM9 molecule (n=11, k=4):**
- ✓ Coefficient of Variation: **0.073** (excellent uniformity)
- ✓ Max/Min frequency ratio: **1.34x**
- ✓ **9.8x better uniformity** than UGS sampler

## Installation

```bash
pip install -e src/uniform_sampler --no-build-isolation
```

## Usage

### Python API

```python
import torch
import uniform_sampler

# Prepare graph data (PyG format)
edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
ptr = torch.tensor([0, 3], dtype=torch.long)  # [B+1] batch pointers

# Sample uniformly
nodes_t, edge_index_t, edge_ptr_t, sample_ptr_t, edge_src_t = \
    uniform_sampler.sample_batch(
        edge_index,      # [2, E] edge connectivity
        ptr,             # [B+1] graph pointers
        m_per_graph=100, # samples per graph
        k=5,             # subgraph size
        mode="sample",   # mode (currently only "sample" supported)
        seed=42          # random seed for reproducibility
    )
```

### Return Values

- `nodes_t`: [total_samples, k] - node IDs for each sample
- `edge_index_t`: [2, total_edges] - edge connectivity
- `edge_ptr_t`: [total_samples+1] - edge offsets for each sample
- `sample_ptr_t`: [B+1] - sample offsets for each graph
- `edge_src_t`: [total_edges] - original edge indices

### Seed Control

```python
# Reproducible sampling
nodes1, *_ = uniform_sampler.sample_batch(edge_index, ptr, m_per_graph=10, k=3, seed=42)
nodes2, *_ = uniform_sampler.sample_batch(edge_index, ptr, m_per_graph=10, k=3, seed=42)
assert torch.equal(nodes1, nodes2)  # ✓ Same results

# Different samples
nodes3, *_ = uniform_sampler.sample_batch(edge_index, ptr, m_per_graph=10, k=3, seed=123)
assert not torch.equal(nodes1, nodes3)  # ✓ Different results
```

## Algorithm

Uses **exhaustive enumeration** approach:

1. **Preprocessing**: Enumerate all connected k-subgraphs in the graph
2. **Sampling**: Uniformly select from enumerated list using RNG

### Complexity

- **Time**: O(C(n,k)) where C(n,k) is the number of connected k-subgraphs
- **Space**: O(C(n,k) * k)

### Feasibility

| Graph Size | k Range | # Subgraphs | Status |
|------------|---------|-------------|---------|
| n ≤ 20 | k ≤ 6 | < 1,000 | ✓ **Excellent** |
| n ≤ 30 | k ≤ 6 | < 10,000 | ✓ **Good** |
| n ≤ 50 | k ≤ 6 | < 100,000 | ⚠️ **Marginal** |
| n > 50 or k > 8 | - | > 1M | ✗ **Not feasible** |

**For QM9 dataset** (n ≤ 20): ✓ Perfect fit!

## Comparison with UGS Sampler

| Metric | Uniform Sampler | UGS Sampler | Improvement |
|--------|----------------|-------------|-------------|
| **Coefficient of Variation** | 0.073 | 0.711 | **9.8x better** |
| **Max/Min Ratio** | 1.34 | 11.72 | **8.7x better** |
| **Uniformity** | Excellent ✓ | Poor ✗ | **Provably uniform** |

## Testing

```bash
# Basic functionality test
python tests/test_uniform_sampler.py

# Comparison with UGS sampler
python tests/test_compare_samplers.py
```

## Limitations

- Only works efficiently for small graphs (n < 50) and small k (k < 8)
- For larger graphs, consider MCMC-based approaches
- Currently CPU-only (enumeration step)
- No caching between calls (re-enumerates each time)

## Future Improvements

- [ ] Caching enumerated subgraphs
- [ ] GPU-accelerated enumeration
- [ ] MCMC sampling for large graphs
- [ ] Batch-aware enumeration
