# SS-GNN Sampler Profiling Tools

This directory contains tools for profiling and analyzing subgraph samplers (RWR, Uniform, UGS, Epsilon-Uniform) on various graph datasets.

## Available Tools

### 1. Sampler Validity & Performance Profiler (`profile_sampler_validity.py`)

Profiles samplers for **validity** and **performance** across entire datasets. Tracks which graphs produce invalid samples (with -1 padding) and measures processing time.

**Use this when you want to:**
- Check how many graphs in a dataset cause invalid sampling
- Measure total processing time for a dataset
- Identify problematic graphs (small graphs that can't produce k-sized samples)
- Benchmark sampler performance

**Usage:**
```bash
# Profile RWR sampler on PROTEINS dataset
python tools/profile_sampler_validity.py --dataset PROTEINS --sampler rwr --k 8 --m 100

# Profile uniform sampler on QM9
python tools/profile_sampler_validity.py --dataset QM9 --sampler uniform --k 6 --m 50

# Profile on MUTAG with custom parameters
python tools/profile_sampler_validity.py --dataset MUTAG --sampler rwr --k 5 --m 200 --seed 42
```

**Key Options:**
- `--dataset NAME`: Dataset name (PROTEINS, QM9, MUTAG, ZINC, etc.)
- `--sampler {rwr,uniform,ugs,epsilon_uniform}`: Sampler to use
- `--k N`: Subgraph size
- `--m N`: Number of samples per graph
- `--p-restart FLOAT`: Restart probability for RWR (default: 0.2)
- `--epsilon FLOAT`: Epsilon for epsilon_uniform (default: 0.1)
- `--seed N`: Random seed (default: 42)

**Output:**
- Total processing time (seconds and minutes)
- Processing rate (graphs/second)
- Number and percentage of graphs with invalid samples
- List of problematic graphs with their sizes
- Statistics on invalid sample counts

---

### 2. Sampler CV Profiler (`profile_sampler_cv.py`)

Profiles samplers for **sampling uniformity** by calculating the coefficient of variation (CV) against exact k-graphlet enumeration.

**Use this when you want to:**
- Measure how uniform the sampler's distribution is
- Compare different samplers' uniformity
- Understand sampler bias toward certain subgraphs
- Analyze coverage of the sample space

**Usage:**
```bash
# Profile RWR sampler on PROTEINS with multiple k values
python tools/profile_sampler_cv.py --dataset PROTEINS --sampler rwr --k 5 6 7

# Profile uniform sampler on QM9
python tools/profile_sampler_cv.py --dataset QM9 --sampler uniform --k 6

# Use dataset-specific default k values
python tools/profile_sampler_cv.py --dataset MUTAG --sampler rwr

# Control graph size selection
python tools/profile_sampler_cv.py --dataset QM9 --sampler rwr --min-nodes 15 --max-nodes 25
```

**Key Options:**
- `--dataset NAME`: Dataset name (PROTEINS, QM9, MUTAG, ZINC, etc.)
- `--sampler {rwr,uniform,ugs,epsilon_uniform}`: Sampler to use
- `--k K [K ...]`: List of k values to test (e.g., `--k 4 5 6`)
- `--num-samples N`: Number of samples to draw (default: 3000)
- `--min-nodes N`: Min graph size for selection (default: 10)
- `--max-nodes N`: Max graph size for selection (default: 500)
- `--max-graphlets N`: Skip if too many graphlets (default: 50000)
- `--no-parallel`: Disable parallel enumeration

**Output:**
- Exact k-graphlet enumeration time and count
- Sampling time and valid sample count
- **Coefficient of Variation (CV)**: std/mean of frequencies (lower = more uniform)
- Coverage: percentage of all graphlets that were sampled
- Frequency statistics and top-5 most frequent graphlets

---

## Supported Datasets

Both tools support the following datasets:

### TUDataset
- **PROTEINS**: Protein structures (1113 graphs, ~39 nodes avg)
- **MUTAG**: Mutagenic compounds (188 graphs, ~18 nodes avg)
- **ENZYMES**: Enzyme structures (600 graphs, ~32 nodes avg)
- **DD**: Protein structures (1178 graphs, ~284 nodes avg)
- And many more TUDatasets...

### PyG Datasets
- **QM9**: Molecular graphs (~130k molecules, ~18 atoms avg)
- **ZINC**: Molecular graphs (subset: 10k, full: 250k)
- **Peptides-func**: Peptide structures (LRGB benchmark)

To add a new TUDataset, simply use its name (e.g., `--dataset COLLAB`).

---

## Supported Samplers

All tools support the following samplers:
- **rwr**: Random Walk with Restart (configurable p_restart)
- **uniform**: Uniform subgraph sampler
- **ugs**: Universal Graph Sampler
- **epsilon_uniform**: Epsilon-approximate uniform sampler (configurable epsilon)

---

## Comparison: Which Tool to Use?

| Goal | Use This Tool |
|------|---------------|
| Check for invalid samples in dataset | `profile_sampler_validity.py` |
| Measure total processing time | `profile_sampler_validity.py` |
| Find problematic graphs | `profile_sampler_validity.py` |
| Measure sampling uniformity | `profile_sampler_cv.py` |
| Compare sampler bias | `profile_sampler_cv.py` |
| Understand sample distribution | `profile_sampler_cv.py` |

---

## Example Workflows

### Workflow 1: Validate Sampler on New Dataset

```bash
# Step 1: Check for invalid samples across entire dataset
python tools/profile_sampler_validity.py --dataset PROTEINS --sampler rwr --k 8 --m 100

# Step 2: If issues found, analyze uniformity on a representative graph
python tools/profile_sampler_cv.py --dataset PROTEINS --sampler rwr --k 8 --num-samples 5000
```

### Workflow 2: Compare Multiple Samplers

```bash
# Terminal 1: Profile RWR sampler
python tools/profile_sampler_cv.py --dataset PROTEINS --sampler rwr --k 5 6 7

# Terminal 2: Profile Uniform sampler
python tools/profile_sampler_cv.py --dataset PROTEINS --sampler uniform --k 5 6 7

# Terminal 3: Profile UGS sampler
python tools/profile_sampler_cv.py --dataset PROTEINS --sampler ugs --k 5 6 7

# Compare the CV values - lower CV = more uniform
```

### Workflow 3: Tune Sampler Parameters

```bash
# Test different p_restart values for RWR sampler
python tools/profile_sampler_cv.py --dataset PROTEINS --sampler rwr --k 6 --p-restart 0.1
python tools/profile_sampler_cv.py --dataset PROTEINS --sampler rwr --k 6 --p-restart 0.2
python tools/profile_sampler_cv.py --dataset PROTEINS --sampler rwr --k 6 --p-restart 0.3

# Compare CV and coverage to find optimal p_restart
```

---

## Performance Notes

### `profile_sampler_validity.py`
- Processes entire datasets (can take minutes for large datasets)
- Adaptive progress reporting (prints ~10 updates regardless of dataset size)
- Lightweight - only checks for -1 nodes, no heavy computation

### `profile_sampler_cv.py`
- Runs exact k-graphlet enumeration (computationally expensive)
- Uses parallel processing by default (speeds up 2-8x)
- Practical limits: k ≤ 8, graph size ≤ 100 nodes
- Selects a single representative graph from the dataset

---

## Tips

1. **Start with validity profiling** before analyzing uniformity
2. **Use smaller k values** (4-6) for faster experimentation
3. **Increase `--num-samples`** in CV profiling for more accurate estimates
4. **Enable parallel enumeration** for faster CV profiling (enabled by default)
5. **Use different seeds** when running parallel experiments
6. **Adjust graph size range** in CV profiling to target specific graph sizes

---

## Additional Documentation

- [Detailed CV Profiler Documentation](README_profile_sampler.md)

---

## Requirements

- Python 3.7+
- PyTorch
- PyTorch Geometric
- NumPy
- Compiled sampler extensions (rwr_sampler, uniform_sampler, etc.)

Install datasets:
```bash
# Datasets will be automatically downloaded on first use
# Data will be stored in data/TUDataset, data/QM9, data/ZINC, etc.
```
