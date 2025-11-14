# test_rwr_profiler.py
import time
import random
import numpy as np
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Batch

# Try to import sampler extension (rwr_sampler preferred, fallback to uniform_sampler)
try:
    import rwr_sampler as sampler_ext
except Exception:
    try:
        import uniform_sampler as sampler_ext
    except Exception:
        raise ImportError("Could not import rwr_sampler or uniform_sampler extension. Build/install it first.")

def build_batch_from_list(data_list, device='cpu'):
    batch = Batch.from_data_list(data_list).to(device)
    edge_index = batch.edge_index.long()
    ptr = batch.ptr.long()
    return edge_index, ptr, batch

def time_sample_batch(edge_index, ptr, m_per_graph, k, **kwargs):
    torch.cuda.synchronize() if edge_index.is_cuda else None
    t0 = time.perf_counter()
    out = sampler_ext.sample_batch(edge_index, ptr, m_per_graph, k, **kwargs)
    torch.cuda.synchronize() if edge_index.is_cuda else None
    t1 = time.perf_counter()
    return (t1 - t0), out

def main():
    dataset = TUDataset(root='data/TUDataset', name='PROTEINS')
    num_graphs = len(dataset)
    print(f"Loaded PROTEINS: {num_graphs} graphs")

    # Parameters
    k = 6
    m_per_graph = 50       # number of subgraphs sampled per graph in the 32-graph batch (tune)
    p_restart = 0.2
    mode = "sample"
    seed = 42
    device = 'cpu'   # change to 'cuda' if desired and extension supports CUDA transfer

    # ---- 1) measure time for a batch of 32 graphs ----
    B = 32
    if num_graphs < B:
        raise RuntimeError(f"PROTEINS has only {num_graphs} graphs, need at least {B}")
    # choose first 32 graphs (or random)
    list32 = [dataset[i] for i in range(B)]
    edge_index_32, ptr_32, batch32 = build_batch_from_list(list32, device=device)

    t_elapsed, _ = time_sample_batch(
        edge_index_32, ptr_32, m_per_graph, k,
        mode=mode, seed=seed, p_restart=p_restart
    )
    print(f"Time to sample m={m_per_graph} subgraphs per graph on B={B} graphs (k={k}): {t_elapsed:.6f} s")

    # ---- 2) CV on a random single graph ----
    runs = 100
    rand_idx = random.randrange(num_graphs)
    data_rand = dataset[rand_idx]
    edge_index_1, ptr_1, batch1 = build_batch_from_list([data_rand], device=device)

    times = []
    # Warm-up runs (optional)
    for _ in range(5):
        _t, _ = time_sample_batch(edge_index_1, ptr_1, m_per_graph, k, mode=mode, seed=seed, p_restart=p_restart)

    for i in range(runs):
        t0, _ = time_sample_batch(edge_index_1, ptr_1, m_per_graph, k, mode=mode, seed=seed + i, p_restart=p_restart)
        times.append(t0)

    times = np.array(times)
    mean_t = times.mean()
    std_t = times.std(ddof=1)
    cv = std_t / mean_t if mean_t != 0 else float('nan')

    print(f"Random graph index: {rand_idx}, nodes: {data_rand.num_nodes}, edges: {data_rand.num_edges}")
    print(f"Single-graph sampling (m={m_per_graph}, k={k}) over {runs} runs:")
    print(f"  mean = {mean_t:.6f} s, std = {std_t:.6f} s, CV = {cv:.4f}")

    # Optionally, print percentiles
    pcts = np.percentile(times, [5, 25, 50, 75, 95])
    print("  time percentiles (s): 5%,25%,50%,75%,95% ->", np.round(pcts, 6))

if __name__ == "__main__":
    main()
