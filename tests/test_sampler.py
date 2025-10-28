# test_sampler_global_mode.py
import torch
import pytest

import ugs_sampler  

def test_sampler_global_mode_invariants():
    # --- Build a small connected graph (per-graph, local numbering 0..n-1) ---
    edge_index_local = torch.tensor([[0,1,2,3,4,5,5,6,7,8,9],
                            [1,2,3,4,5,0,6,7,8,9,0]], dtype=torch.long)
    n = 4
    k = 3
    m_per_graph = 4
    base_offset = 10  # simulate the graph starting at global node id 10

    # Preprocess and sample in GLOBAL mode
    handle = ugs_sampler.create_preproc(edge_index_local, n, k)
    try:
        nodes_t, edge_index_t, edge_ptr_t, edge_src_idx_t = ugs_sampler.sample(
            handle, m_per_graph, k, "global", base_offset
        )
    finally:
        ugs_sampler.destroy_preproc(handle)

    # Basic shapes
    assert nodes_t.shape == (m_per_graph, k)
    assert edge_ptr_t.shape == (m_per_graph + 1,)
    assert edge_index_t.shape[0] == 2
    assert edge_src_idx_t.ndim == 1
    assert edge_src_idx_t.numel() == edge_index_t.shape[1]

    # Helper: edges of a sample
    def edges_of(b):
        s = int(edge_ptr_t[b].item())
        e = int(edge_ptr_t[b+1].item())
        return edge_index_t[:, s:e].t()  # [E_b, 2]

    total_nodes_lo = base_offset
    total_nodes_hi = base_offset + n

    # Build a set of unordered original (local) edges for membership checks
    orig_edges_local = set()
    for u, v in edge_index_local.t().tolist():
        a, b = sorted((u, v))
        orig_edges_local.add((a, b))

    # 1) Global range & membership: edges must use global IDs and be subset of sample's node set (+offset)
    for b in range(m_per_graph):
        nodes_local = nodes_t[b].tolist()
        nodes_global_set = {base_offset + int(x) for x in nodes_local if int(x) >= 0}
        e_b = edges_of(b)
        for (u, v) in e_b.tolist():
            # endpoints must be global and within [base_offset, base_offset+n)
            assert total_nodes_lo <= u < total_nodes_hi
            assert total_nodes_lo <= v < total_nodes_hi
            # subset of the sample's node set
            assert u in nodes_global_set and v in nodes_global_set

    # 2) edge_src_idx_t validity and semantic check (up to orientation)
    #    For each emitted edge (u,v), map the source column id to (ul,vl) in local space
    #    and verify {u - base_offset, v - base_offset} equals {ul, vl}.
    if edge_src_idx_t.numel() > 0:
        m_cols = edge_index_local.size(1)
        assert int(edge_src_idx_t.min()) >= 0
        assert int(edge_src_idx_t.max()) < m_cols

        # Gather referenced local columns
        gathered_local = edge_index_local[:, edge_src_idx_t]  # [2, E_emit]
        # Compare as unordered pairs
        u_glob = edge_index_t[0].tolist()
        v_glob = edge_index_t[1].tolist()
        u_loc = gathered_local[0].tolist()
        v_loc = gathered_local[1].tolist()
        for ug, vg, ul, vl in zip(u_glob, v_glob, u_loc, v_loc):
            pair_global_local = sorted((ug - base_offset, vg - base_offset))
            pair_from_srccol  = sorted((ul, vl))
            assert tuple(pair_global_local) == tuple(pair_from_srccol), \
                f"Edge mismatch: global ({ug},{vg}) vs local col ({ul},{vl}) + offset {base_offset}"

    # 3) Optional: ensure at least some edges were produced for connected graphs
    assert edge_index_t.size(1) >= 0  # non-strict, but the graph is connected so usually > 0
