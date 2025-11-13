#!/usr/bin/env python3
"""Test uniform_sampler with larger QM9 molecules that can produce size-8 subgraphs"""
import torch
import uniform_sampler
import time
import numpy as np
from torch_geometric.datasets import QM9

print("=" * 70)
print("Testing UNIFORM SAMPLER - Larger QM9 Molecules (≥15 nodes)")
print("=" * 70)
print()

# Load QM9 dataset
print("Loading QM9 dataset...")
dataset = QM9(root='./data/QM9')

# Filter for larger molecules (at least 15 nodes to ensure k=8 subgraphs exist)
MIN_NODES = 15
print(f"Filtering molecules with at least {MIN_NODES} nodes...")
large_molecules = []
for i in range(len(dataset)):
    if dataset[i].num_nodes >= MIN_NODES:
        large_molecules.append(i)
        if len(large_molecules) >= 128:
            break

print(f"Found {len(large_molecules)} large molecules")
print()

# Test parameters
BATCH_SIZE = 128
NUM_SUBGRAPHS = 50
SUBGRAPH_SIZE = 8

print("Test Configuration:")
print("-" * 70)
print(f"  Batch size: {BATCH_SIZE} graphs")
print(f"  Subgraphs per graph: {NUM_SUBGRAPHS}")
print(f"  Subgraph size (k): {SUBGRAPH_SIZE}")
print(f"  Minimum nodes per graph: {MIN_NODES}")
print()

# Load batch
batch_data = [dataset[i] for i in large_molecules[:BATCH_SIZE]]

# Get statistics
node_counts = [data.num_nodes for data in batch_data]
edge_counts = [data.edge_index.size(1) for data in batch_data]

print(f"Graph statistics in batch:")
print(f"  Nodes per graph - Mean: {np.mean(node_counts):.1f}, "
      f"Min: {min(node_counts)}, Max: {max(node_counts)}")
print(f"  Edges per graph - Mean: {np.mean(edge_counts):.1f}, "
      f"Min: {min(edge_counts)}, Max: {max(edge_counts)}")
print()

# Prepare batch
edge_index_list = []
ptr = [0]
offset = 0

for data in batch_data:
    edge_index_list.append(data.edge_index + offset)
    offset += data.num_nodes
    ptr.append(offset)

edge_index = torch.cat(edge_index_list, dim=1)
ptr = torch.tensor(ptr, dtype=torch.long)

print(f"Batched graph:")
print(f"  Total nodes: {ptr[-1]}")
print(f"  Total edges: {edge_index.size(1)}")
print()

# Benchmark
print("Running benchmark...")
print("-" * 70)

# Warm-up
_ = uniform_sampler.sample_batch(edge_index, ptr,
                                m_per_graph=NUM_SUBGRAPHS, k=SUBGRAPH_SIZE, seed=42)

# Actual runs
NUM_RUNS = 10
times = []

for run in range(NUM_RUNS):
    start_time = time.time()
    nodes_t, edge_index_t, edge_ptr_t, sample_ptr_t, edge_src_t = \
        uniform_sampler.sample_batch(edge_index, ptr,
                                    m_per_graph=NUM_SUBGRAPHS, k=SUBGRAPH_SIZE, seed=42+run)
    elapsed = time.time() - start_time
    times.append(elapsed)

mean_time = np.mean(times) * 1000
std_time = np.std(times) * 1000

print(f"Timing over {NUM_RUNS} runs:")
print(f"  Mean: {mean_time:.2f} ± {std_time:.2f} ms")
print(f"  Min:  {np.min(times)*1000:.2f} ms")
print(f"  Max:  {np.max(times)*1000:.2f} ms")
print()

# Verify completeness
complete_count = 0
incomplete_count = 0

for i in range(BATCH_SIZE * NUM_SUBGRAPHS):
    nodes = nodes_t[i]
    valid_nodes = (nodes >= 0).sum().item()
    if valid_nodes == SUBGRAPH_SIZE:
        complete_count += 1
    else:
        incomplete_count += 1

total_subgraphs = BATCH_SIZE * NUM_SUBGRAPHS
complete_pct = 100 * complete_count / total_subgraphs

print("Completeness verification:")
print("-" * 70)
print(f"  Complete subgraphs: {complete_count}/{total_subgraphs} ({complete_pct:.2f}%)")
print(f"  Incomplete subgraphs: {incomplete_count}/{total_subgraphs}")

if complete_pct >= 99:
    print(f"  ✓ EXCELLENT: {complete_pct:.2f}% of subgraphs are complete!")
elif complete_pct >= 90:
    print(f"  ✓ GOOD: {complete_pct:.2f}% of subgraphs are complete")
else:
    print(f"  ⚠️  {complete_pct:.2f}% completeness")

print()

# Check individual graphs
print("Per-graph completeness check:")
print("-" * 70)
graphs_with_all_complete = 0

for i in range(BATCH_SIZE):
    start_idx = sample_ptr_t[i].item()
    end_idx = sample_ptr_t[i+1].item()

    graph_complete = 0
    for j in range(start_idx, end_idx):
        if (nodes_t[j] >= 0).sum().item() == SUBGRAPH_SIZE:
            graph_complete += 1

    if graph_complete == NUM_SUBGRAPHS:
        graphs_with_all_complete += 1

print(f"  Graphs with all {NUM_SUBGRAPHS} complete subgraphs: {graphs_with_all_complete}/{BATCH_SIZE}")
print(f"  Percentage: {100*graphs_with_all_complete/BATCH_SIZE:.1f}%")

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"For larger molecules (≥{MIN_NODES} nodes):")
print(f"  Sampling time: {mean_time:.2f} ms for {BATCH_SIZE} graphs")
print(f"  {complete_pct:.1f}% of subgraphs are complete (size {SUBGRAPH_SIZE})")
print(f"  Throughput: {BATCH_SIZE/(mean_time/1000):.1f} graphs/sec")
print("=" * 70)
