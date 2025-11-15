# Sampler Profiling Tool - Coefficient of Variation (CV)

Profile subgraph samplers by calculating the coefficient of variation (CV) of their empirical distribution against exact k-graphlet enumeration.

## Features

- **Exact k-graphlet enumeration** using optimized BFS/DFS (much faster than brute-force)
- **Parallel enumeration** across all CPU cores
- **Multiple sampler support**: rwr, uniform, ugs, epsilon_uniform
- **Multiple datasets**: PROTEINS, QM9, MUTAG, ZINC, Peptides-func (LRGB), and other TUDatasets
- **Comprehensive profiling**: timing, CV, coverage, frequency distributions

## Usage

```bash
# Profile RWR sampler on PROTEINS with k=5,6,7
python tools/profile_sampler_cv.py --dataset PROTEINS --sampler rwr --k 5 6 7

# Profile uniform sampler on QM9 with k=6
python tools/profile_sampler_cv.py --dataset QM9 --sampler uniform --k 6

# Profile on MUTAG with default k values (dataset-specific)
python tools/profile_sampler_cv.py --dataset MUTAG --sampler rwr

# Profile with custom number of samples and seed
python tools/profile_sampler_cv.py --dataset PROTEINS --sampler rwr --k 5 6 --num-samples 5000 --seed 123

# Profile UGS sampler on ZINC without parallel enumeration
python tools/profile_sampler_cv.py --dataset ZINC --sampler ugs --k 4 5 --no-parallel

# Control graph size selection
python tools/profile_sampler_cv.py --dataset QM9 --sampler rwr --min-nodes 15 --max-nodes 25
```

## Options

- `--dataset NAME`: **[REQUIRED]** Dataset name (e.g., PROTEINS, QM9, MUTAG, ZINC)
- `--sampler {rwr,uniform,ugs,epsilon_uniform}`: Choose which sampler to profile (default: rwr)
- `--k K [K ...]`: List of k values to test (e.g., `--k 4 5 6 7`). Uses dataset-specific defaults if not specified
- `--num-samples N`: Number of subgraphs to sample per graph (default: 3000)
- `--min-nodes N`: Minimum number of nodes for graph selection (default: 10)
- `--max-nodes N`: Maximum number of nodes for graph selection (default: 500)
- `--max-graphlets N`: Skip sampling if exact enumeration finds more than N graphlets (default: 50000)
- `--seed N`: Random seed for reproducibility (default: 42)
- `--p-restart FLOAT`: Restart probability for RWR sampler (default: 0.2)
- `--epsilon FLOAT`: Epsilon parameter for epsilon_uniform sampler (default: 0.1)
- `--no-parallel`: Disable parallel enumeration (useful for debugging)

## Dataset-Specific Default k Values

- **MUTAG, ZINC**: k=[4, 5, 6]
- **PROTEINS, QM9**: k=[5, 6, 7]
- **Others**: k=[5, 6, 7]

## Output

For each k value, the tool reports:

1. **Exact enumeration**:
   - Time taken
   - Number of distinct k-graphlets found

2. **Sampling**:
   - Time taken
   - Number of valid samples
   - CV (coefficient of variation)
   - Coverage (% of distinct graphlets sampled)
   - Frequency statistics (min, max, median)
   - Top-5 most frequent graphlets
   - Time ratio (sampling/exact)

## Coefficient of Variation (CV)

- **CV = std / mean** of graphlet frequencies
- **Lower CV** → more uniform sampling
- **CV = 0** → perfect uniformity (only possible if 1 unique graphlet)
- **Higher CV** → more biased sampling

## Performance Notes

- **Exact enumeration** uses DFS from each node (avoids disconnected combinations)
- **Parallelization** speeds up enumeration significantly for graphs with >20 nodes
- **Practical limits**: k ≤ 8, n ≤ 100 for reasonable runtime
- For larger graphs, exact enumeration may timeout (>100k graphlets)

## Example Output

```
======================================================================
SAMPLER CV PROFILER: RWR
Testing CV with exact k-graphlet enumeration
======================================================================
Dataset: PROTEINS
Parallel enumeration: ENABLED (8 CPUs)
k values: [5, 6, 7]
Samples per graph: 3000
Graph size range: 10-500 nodes
Random seed: 42
p_restart: 0.2
======================================================================

======================================================================
Dataset: PROTEINS
======================================================================
Loading PROTEINS dataset...
Loaded 1113 graphs from PROTEINS

Selected graph 42:
  Nodes: 28, Edges: 45

--- k = 5 ---
1. Exact enumeration...
   Time: 8.2341 s
   Found 145 distinct 5-graphlets
2. RWR sampling...
   Time: 0.0312 s
   Valid samples: 2998/3000
3. CV = 2.145327
   Unique in samples: 138
   Total distinct (exact): 145
   Coverage: 95.17%
   Frequency stats: min=1, max=285, median=12.0
   Top-5 most frequent:
     (4, 5, 6, 17, 18): 285 (9.51%)
     (0, 1, 10, 13, 14): 198 (6.61%)
     ...
   Time ratio (sampling/exact): 0.00x
```

## Parallel Profiling

To profile multiple samplers in parallel, run them in separate terminals:

```bash
# Terminal 1 - RWR sampler on PROTEINS
python tools/profile_sampler_cv.py --dataset PROTEINS --sampler rwr --k 5 6 7

# Terminal 2 - Uniform sampler on PROTEINS
python tools/profile_sampler_cv.py --dataset PROTEINS --sampler uniform --k 5 6 7

# Terminal 3 - Compare on different dataset (QM9)
python tools/profile_sampler_cv.py --dataset QM9 --sampler rwr --k 5 6 7
```

Compare the CV values to evaluate uniformity across samplers.

## Tips

- **Increase `--num-samples`** for more accurate CV estimates (but slower)
- **Use different `--seed`** values when running parallel experiments to avoid interference
- **Lower k values** (4-6) run faster than higher ones (7-8)
- **Disable `--no-parallel`** if you want to debug or reduce CPU usage
