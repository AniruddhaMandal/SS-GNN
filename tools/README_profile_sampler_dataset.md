# Sampler Coverage Profiler for Entire Datasets

## Overview

`profile_sampler_dataset.py` is a comprehensive tool for evaluating subgraph sampler performance across entire datasets. It helps you determine:

1. **Which sampler** provides the best coverage for your dataset
2. **Which k value** is appropriate (balance between coverage and computational cost)
3. **How many samples per graph (m)** are needed to achieve good coverage

## What Does "Coverage" Mean?

**Coverage** is the fraction of all exact k-graphlets that appear in your samples:

```
Coverage = (Number of unique exact k-graphlets found in samples) / (Total number of exact k-graphlets)
```

- **Coverage = 1.0 (100%)**: Perfect! Your samples include every possible k-graphlet at least once
- **Coverage = 0.8 (80%)**: Good. Your samples cover 80% of all possible k-graphlets
- **Coverage = 0.5 (50%)**: Fair. Half of the k-graphlets are represented in your samples
- **Coverage < 0.5**: Poor. Many k-graphlets are missing from your samples

**A good sampler** achieves high coverage with a reasonable number of samples (m).

## How It Works

For each graph in the dataset:

1. **Enumerate exact k-graphlets** using efficient BFS/DFS (not brute force)
2. **Sample m subgraphs** using the specified sampler
3. **Calculate coverage**: what % of exact k-graphlets appear in the samples
4. **Skip graphs** where enumeration would be too expensive (> max_graphlets)

The tool reports both per-graph statistics and aggregate statistics across the entire dataset.

## Usage

### Basic Usage

```bash
# Test RWR sampler on PROTEINS dataset with k=5 and m=100
python tools/profile_sampler_dataset.py --dataset PROTEINS --sampler rwr --k 5 --m 100

# Test multiple k values and m values
python tools/profile_sampler_dataset.py --dataset PROTEINS --sampler rwr --k 5 6 7 --m 50 100 200

# Compare multiple samplers
python tools/profile_sampler_dataset.py --dataset PROTEINS --sampler rwr uniform ugs --k 5 --m 100
```

### Advanced Usage

```bash
# Save results to CSV for later analysis
python tools/profile_sampler_dataset.py --dataset PROTEINS --sampler rwr --k 5 --m 100 --output results/proteins_rwr.csv

# Process only first 100 graphs (useful for large datasets)
python tools/profile_sampler_dataset.py --dataset QM9 --sampler rwr --k 6 --m 100 --max-graphs 100

# Increase max_graphlets threshold for larger graphs
python tools/profile_sampler_dataset.py --dataset PROTEINS --sampler rwr --k 6 --m 100 --max-graphlets 100000

# Disable parallel processing (for debugging)
python tools/profile_sampler_dataset.py --dataset MUTAG --sampler rwr --k 4 --m 100 --no-parallel
```

### Sampler-Specific Parameters

```bash
# RWR sampler: adjust restart probability
python tools/profile_sampler_dataset.py --dataset PROTEINS --sampler rwr --k 5 --m 100 --p-restart 0.3

# Epsilon-uniform sampler: adjust epsilon
python tools/profile_sampler_dataset.py --dataset PROTEINS --sampler epsilon_uniform --k 5 --m 100 --epsilon 0.2
```

## Command-Line Arguments

| Argument | Required | Description | Example |
|----------|----------|-------------|---------|
| `--dataset` | Yes | Dataset name | `PROTEINS`, `MUTAG`, `QM9`, `ZINC` |
| `--sampler` | Yes | Sampler(s) to test | `rwr`, `uniform`, `ugs`, `epsilon_uniform` |
| `--k` | Yes | k value(s) for subgraph size | `--k 5 6 7` |
| `--m` | Yes | Number of samples per graph | `--m 50 100 200` |
| `--max-graphs` | No | Limit number of graphs | `--max-graphs 100` |
| `--max-graphlets` | No | Skip if more than this many exact graphlets | `--max-graphlets 50000` (default) |
| `--output` | No | Save results to CSV | `--output results.csv` |
| `--seed` | No | Random seed | `--seed 42` (default) |
| `--p-restart` | No | RWR restart probability | `--p-restart 0.2` (default) |
| `--epsilon` | No | Epsilon-uniform parameter | `--epsilon 0.1` (default) |
| `--no-parallel` | No | Disable parallel processing | (flag) |

## Interpreting Results

### Console Output

The tool prints detailed statistics for each (k, m) combination:

```
================================================================================
Processing: k=5, m=100
================================================================================
  Processed 50/100 graphs... (completed: 45, skipped: 5)

Summary for k=5, m=100:
  Completed: 95/100
  Skipped (k too large): 2
  Skipped (too many graphlets): 3

  Coverage statistics:
    Mean: 0.7532
    Median: 0.7891
    Std: 0.1245
    Min: 0.3421
    Max: 0.9876

  Exact graphlets per graph:
    Mean: 1234.5
    Median: 987.0
    Min: 145
    Max: 4523

  Graphs with coverage < 50%: 5
    Graph 12: 0.3421
    Graph 45: 0.4123
    ...
```

### CSV Output

When you use `--output`, results are saved to CSV with these columns:

| Column | Description |
|--------|-------------|
| `dataset` | Dataset name |
| `sampler` | Sampler name |
| `graph_idx` | Graph index in dataset |
| `num_nodes` | Number of nodes in graph |
| `num_edges` | Number of edges in graph |
| `k` | Subgraph size |
| `m` | Number of samples per graph |
| `status` | `completed`, `skipped_k_too_large`, or `skipped_too_many_graphlets` |
| `exact_graphlets` | Total number of exact k-graphlets |
| `covered_graphlets` | Number of exact k-graphlets found in samples |
| `valid_samples` | Number of valid samples (should be ≈ m) |
| `coverage` | Coverage fraction (0.0 to 1.0) |
| `time_exact` | Time for exact enumeration (seconds) |
| `time_sample` | Time for sampling (seconds) |

You can load this CSV in Python/R/Excel for further analysis.

## How to Choose Parameters

### 1. Choosing the Best Sampler

Run multiple samplers with the same k and m:

```bash
python tools/profile_sampler_dataset.py --dataset PROTEINS --sampler rwr uniform ugs --k 5 --m 100
```

Compare average coverage:
- **Best sampler** = highest average coverage
- Also consider variance: lower variance = more consistent across graphs

### 2. Choosing the Best k

Run multiple k values with the same sampler and m:

```bash
python tools/profile_sampler_dataset.py --dataset PROTEINS --sampler rwr --k 4 5 6 7 --m 100
```

Observe:
- **Smaller k** = easier to achieve high coverage, but less expressive features
- **Larger k** = more expressive features, but harder to achieve high coverage
- Choose k that balances coverage and expressiveness

### 3. Choosing the Best m (samples per graph)

Run multiple m values with the same sampler and k:

```bash
python tools/profile_sampler_dataset.py --dataset PROTEINS --sampler rwr --k 5 --m 50 100 200 500
```

Look for the point of diminishing returns:
- If coverage plateaus (e.g., m=100 and m=200 give similar coverage), use the smaller m
- If coverage keeps increasing, you may need more samples

**Rule of thumb**: Target 80-90% coverage. More samples beyond this point may not help much.

### 4. Example Workflow

```bash
# Step 1: Compare samplers (k=5, m=100)
python tools/profile_sampler_dataset.py --dataset PROTEINS --sampler rwr uniform ugs --k 5 --m 100

# Result: RWR has highest coverage (0.85 vs 0.72 vs 0.68)

# Step 2: Find best k for RWR
python tools/profile_sampler_dataset.py --dataset PROTEINS --sampler rwr --k 4 5 6 7 --m 100

# Result: k=5 gives 0.85 coverage, k=6 gives 0.62 coverage
# Choose k=5 (better coverage)

# Step 3: Find best m for RWR with k=5
python tools/profile_sampler_dataset.py --dataset PROTEINS --sampler rwr --k 5 --m 50 100 200 500

# Result: m=100 gives 0.85, m=200 gives 0.87, m=500 gives 0.88
# Choose m=100 (good coverage with fewer samples)

# Final choice: RWR sampler, k=5, m=100
```

## Performance Tips

1. **Start small**: Test on a subset first using `--max-graphs 100`
2. **Use parallel processing**: Default is enabled (faster), disable only for debugging
3. **Adjust max_graphlets**: If many graphs are skipped, try increasing `--max-graphlets`
4. **Save to CSV**: Use `--output` to save results for later analysis without re-running

## Comparison with profile_sampler_cv.py

| Feature | profile_sampler_cv.py | profile_sampler_dataset.py |
|---------|----------------------|----------------------------|
| Scope | Single random graph | Entire dataset |
| Metric | Coefficient of Variation (CV) | Coverage |
| Purpose | Measure sampling uniformity | Measure sampling completeness |
| Output | Console only | Console + CSV |
| Use case | Deep dive into one graph | Dataset-level evaluation |

**When to use which?**

- Use `profile_sampler_cv.py` when you want to understand sampling uniformity on a specific graph
- Use `profile_sampler_dataset.py` when you want to choose the best sampler/k/m for your entire dataset

## Troubleshooting

### Many graphs are skipped (too many graphlets)

**Problem**: `Skipped (too many graphlets): 50/100`

**Solution**: Increase `--max-graphlets` or use smaller k value

```bash
python tools/profile_sampler_dataset.py --dataset PROTEINS --sampler rwr --k 5 --m 100 --max-graphlets 100000
```

### Coverage is very low (< 0.3)

**Problem**: Sampler not covering enough graphlets

**Solutions**:
1. Increase m (more samples): `--m 200 500 1000`
2. Try a different sampler: `--sampler uniform ugs`
3. Use smaller k: `--k 4` instead of `--k 6`
4. Check if your sampler parameters are appropriate (e.g., `--p-restart`)

### Out of memory

**Problem**: Large graphs or large k causing memory issues

**Solutions**:
1. Process fewer graphs: `--max-graphs 100`
2. Lower `--max-graphlets`: `--max-graphlets 10000`
3. Use smaller k value
4. Disable parallel processing: `--no-parallel`

### Tool runs forever

**Problem**: Exact enumeration taking too long

**Solution**: The tool has a built-in limit (`max_graphlets`). If it's still too slow:
1. Lower `--max-graphlets`: `--max-graphlets 10000`
2. Use smaller k value
3. Process fewer graphs: `--max-graphs 50`

## Example Output

```
================================================================================
PROFILING RWR SAMPLER ON PROTEINS
================================================================================
Number of graphs: 1113 (out of 1113)
k values: [5]
m values: [100]
Max graphlets per graph: 50000
p_restart: 0.2
================================================================================

================================================================================
Processing: k=5, m=100
================================================================================
  Processed 100/1113 graphs... (completed: 95, skipped: 5)
  Processed 200/1113 graphs... (completed: 189, skipped: 11)
  ...
  Processed 1100/1113 graphs... (completed: 1045, skipped: 55)

Summary for k=5, m=100:
  Completed: 1058/1113
  Skipped (k too large): 12
  Skipped (too many graphlets): 43

  Coverage statistics:
    Mean: 0.8234
    Median: 0.8567
    Std: 0.1123
    Min: 0.2341
    Max: 1.0000

  Exact graphlets per graph:
    Mean: 8234.5
    Median: 5678.0
    Min: 120
    Max: 49823

  Graphs with coverage < 50%: 23

================================================================================
FINAL SUMMARY
================================================================================

k=5, m=100: 1058 graphs
  Coverage: 0.8234 ± 0.1123
  Graphs with >90% coverage: 612 (57.8%)
  Graphs with >80% coverage: 784 (74.1%)
  Graphs with >50% coverage: 1035 (97.8%)

================================================================================

Profiling completed!

Notes:
- Coverage = fraction of exact k-graphlets found in samples
- Higher coverage indicates better sampling quality
- Good sampler should achieve high coverage with reasonable m
- Consider trade-off between coverage, k value, and computational cost
================================================================================
```

## References

- Exact k-graphlet enumeration: BFS/DFS-based algorithm (avoids brute force)
- Coverage metric: Standard metric for evaluating sampling completeness
- Related tool: `profile_sampler_cv.py` (measures sampling uniformity)
