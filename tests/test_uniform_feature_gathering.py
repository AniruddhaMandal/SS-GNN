#!/usr/bin/env python3
"""Test that node and edge features are correctly gathered in encode_subgraphs"""
import torch
import uniform_sampler

print("=" * 70)
print("Testing Node and Edge Feature Gathering")
print("=" * 70)
print()

# Create a batch with distinctive node features
edge_index = torch.tensor([
    [0, 1, 1, 2, 2, 0,  3, 4, 4, 5, 5, 3],
    [1, 0, 2, 1, 0, 2,  4, 3, 5, 4, 3, 5],
], dtype=torch.long)
ptr = torch.tensor([0, 3, 6], dtype=torch.long)

# Node features: each node has a unique ID as its feature
# Node 0: [0.0], Node 1: [1.0], Node 2: [2.0], etc.
x = torch.arange(6, dtype=torch.float32).reshape(-1, 1)
print("Node features (x):")
print(x.T)
print()

# Edge attributes: each edge has a unique ID
edge_attr = torch.arange(12, dtype=torch.float32).reshape(-1, 1)
print("Edge attributes (edge_attr):")
print(edge_attr.T)
print()

# Sample
nodes_t, edge_index_t, edge_ptr_t, sample_ptr_t, edge_src_t = \
    uniform_sampler.sample_batch(edge_index, ptr, m_per_graph=2, k=2, mode="sample", seed=42)

print("Sampler output:")
print(f"  nodes_t:\n{nodes_t}")
print(f"  edge_src_t: {edge_src_t}")
print()

# Check 1: Verify nodes_t doesn't contain -1 (padding)
print("Check 1: Padding in nodes_t")
print("-" * 70)
has_padding = (nodes_t == -1).any()
if has_padding:
    print("⚠️  WARNING: nodes_t contains -1 padding!")
    print("   This will cause x[nodes_t.flatten()] to access x[-1] incorrectly!")
    num_padded = (nodes_t == -1).sum().item()
    print(f"   Number of -1 values: {num_padded}")
else:
    print("✓ No padding found in nodes_t")
print()

# Check 2: Simulate feature gathering (like ss_gnn.py line 368-369)
print("Check 2: Node feature gathering")
print("-" * 70)
stacked_nodes = nodes_t.flatten()
print(f"stacked_nodes: {stacked_nodes.tolist()}")

try:
    gathered_x = x[stacked_nodes]
    print(f"gathered_x:\n{gathered_x.T}")

    # Verify correctness: each gathered feature should match the node ID
    for i, node_id in enumerate(stacked_nodes):
        if node_id >= 0:  # Skip padding
            expected = x[node_id]
            actual = gathered_x[i]
            if not torch.allclose(expected, actual):
                print(f"✗ Mismatch at position {i}: expected {expected}, got {actual}")
                break
    else:
        print("✓ Node features correctly gathered")
except IndexError as e:
    print(f"✗ IndexError during feature gathering: {e}")
print()

# Check 3: Edge attribute gathering (like ss_gnn.py line 373)
print("Check 3: Edge feature gathering")
print("-" * 70)
print(f"edge_src_t: {edge_src_t.tolist()}")

try:
    gathered_edge_attr = edge_attr[edge_src_t]
    print(f"gathered_edge_attr:\n{gathered_edge_attr.T}")

    # Verify correctness
    for i, src_idx in enumerate(edge_src_t):
        expected = edge_attr[src_idx]
        actual = gathered_edge_attr[i]
        if not torch.allclose(expected, actual):
            print(f"✗ Mismatch at position {i}: expected {expected}, got {actual}")
            break
    else:
        print("✓ Edge features correctly gathered")
except IndexError as e:
    print(f"✗ IndexError during edge gathering: {e}")
print()

# Check 4: Edge connectivity correctness
print("Check 4: Edge connectivity consistency")
print("-" * 70)
for sample_idx in range(nodes_t.shape[0]):
    nodes = nodes_t[sample_idx]
    edge_start = edge_ptr_t[sample_idx].item()
    edge_end = edge_ptr_t[sample_idx + 1].item()

    if edge_end > edge_start:
        edges_local = edge_index_t[:, edge_start:edge_end]
        edge_srcs = edge_src_t[edge_start:edge_end]

        print(f"Sample {sample_idx}:")
        print(f"  nodes (global IDs): {nodes.tolist()}")
        print(f"  edges (local [0,k-1]): {edges_local.T.tolist()}")
        print(f"  edge sources in batch: {edge_srcs.tolist()}")

        # Verify each edge
        for i in range(edges_local.shape[1]):
            u_local, v_local = edges_local[0, i].item(), edges_local[1, i].item()
            u_global, v_global = nodes[u_local].item(), nodes[v_local].item()
            src_idx = edge_srcs[i].item()

            # Check if this edge exists in original edge_index
            orig_u, orig_v = edge_index[0, src_idx].item(), edge_index[1, src_idx].item()

            # Edge should match (possibly reversed due to undirected)
            if {u_global, v_global} == {orig_u, orig_v}:
                print(f"  ✓ Edge {i}: ({u_global},{v_global}) matches source edge ({orig_u},{orig_v})")
            else:
                print(f"  ✗ Edge {i}: ({u_global},{v_global}) does NOT match source ({orig_u},{orig_v})")

print()
print("=" * 70)
print("Feature Gathering Test Complete")
print("=" * 70)
