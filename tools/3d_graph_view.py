import os
import json
import math
import random
import numpy as np
import networkx as nx
import torch_geometric as pyg
import ugs_sampler
import matplotlib
import matplotlib.pyplot as plt

def nx_to_json(G):
    nodes = []
    for n, data in G.nodes(data=True):
        node_entry = {"id": str(n)}
        if "group" in data:
            node_entry["group"] = data["group"]
        if "color" in data:
            node_entry["color"] = data["color"]
        nodes.append(node_entry)
    links = [{"source": str(u), "target": str(v)} for u, v in G.edges()]
    data = {"nodes": nodes, "links": links}
    with open("graph_json_data/graph.json", "w") as f:
        json.dump(data, f, indent=2)
    print("Exported with", len(nodes), "nodes and", len(links), "links")

def pyg_json_dum(g):
    features = g.x.numpy()
    unique_features, idx = np.unique(features,return_inverse=True,axis=0)
    cmp = plt.get_cmap("tab20")
    color_table = {i: matplotlib.colors.to_hex(cmp(i)) for i in range(len(unique_features))}
    node_colors = {i: color_table[idx[i]] for i in range(len(idx))}
    G = pyg.utils.to_networkx(g)
    nx.set_node_attributes(G,node_colors,"color")
    nx_to_json(G)

def random_peptide_draw():
    data = pyg.datasets.LRGBDataset("../data","Peptides-func")
    i = random.randint(0,len(data))
    pyg_json_dum(data[i])

def export_graphs_to_combined_json(graphs, outpath="multi_graphs.json",
                                   gap=8.0, scale=1.0, seed=42):
    """
    graphs: list of networkx graphs (each may have node attrs e.g. 'color' or 'group')
    gap: spacing between graph centers on the grid
    scale: multiplies spring layout coordinates
    """
    n = len(graphs)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    all_nodes = []
    all_links = []

    for i, G in enumerate(graphs):
        # 3D spring layout for this graph
        pos = nx.spring_layout(G, dim=3, seed=seed+i)  # dict: node -> (x,y,z)

        # compute grid offset for graph i
        col = i % cols
        row = i // cols
        offset = (col * gap, row * gap, 0)

        for node, p in pos.items():
            x, y, z = p[0]*scale + offset[0], p[1]*scale + offset[1], p[2]*scale + offset[2]
            node_id = f"g{i}_{node}"   # unique across all graphs
            node_entry = {
                "id": node_id,
                "orig_id": str(node),     # optional: original id
                "graph_id": i,
                "x": x, "y": y, "z": z
            }
            # pass-through attrs if present
            data = G.nodes[node]
            if "color" in data:
                node_entry["color"] = data["color"]
            if "group" in data:
                node_entry["group"] = data["group"]
            if "label" in data:
                node_entry["label"] = data["label"]
            all_nodes.append(node_entry)

        for u, v in G.edges():
            link = {
                "source": f"g{i}_{u}",
                "target": f"g{i}_{v}",
                "graph_id": i
            }
            # preserve edge attributes like weight/color if present
            data = G.get_edge_data(u, v, default={})
            if "color" in data:
                link["color"] = data["color"]
            if "weight" in data:
                link["weight"] = data["weight"]
            all_links.append(link)

    data = {"nodes": all_nodes, "links": all_links}
    with open(outpath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Exported {len(all_nodes)} nodes and {len(all_links)} links to {outpath}")

def nx_subgraphs(g,k):
    node_num = g.x.size(0)
    handle = ugs_sampler.create_preproc(g.edge_index, node_num, k)
    nodes, edge_index_s, edge_ptr, graph_id = ugs_sampler.sample(handle, m_per_graph=40, k=k)
    B = nodes.size(0)
    
    try:
        features = g.x.numpy()
        unique_features, idx = np.unique(features,return_inverse=True,axis=0)
        cmp = plt.get_cmap("tab20")
        color_table = {i: matplotlib.colors.to_hex(cmp(i)) for i in range(len(unique_features))}
        node_color = {i: color_table[idx[i]] for i in range(len(idx))}
    except:
        print("no color")
        node_color = None

    subgraphs = []
    for b in range(B):
        global_nodes = nodes[b].tolist()
        valid = [i for i,v in enumerate(global_nodes) if v>=0]
        mapping = {i: global_nodes[i] for i in valid}

        start, end = edge_ptr[b].item(), edge_ptr[b+1].item()
        local_edges = edge_index_s[:,start:end].T.tolist()

        G = nx.Graph()
        G.add_nodes_from(mapping.values())
        for u,v in local_edges:
            G.add_edge(mapping[u], mapping[v])
        
        if node_color != None:
            node_color_s = {mapping[i]:node_color[i] for i in mapping.keys()}
        else:
            node_color_s = ["lightblue" for i in mapping.values()]
        
        nx.set_node_attributes(G, node_color_s, "color")
        subgraphs.append(G)

    return subgraphs


if __name__ == "__main__":
    os.makedirs("graph_json_data",exist_ok=True)
    random_peptide_draw()

    data = pyg.datasets.LRGBDataset("../data","Peptides-func")
    i = random.randint(0,len(data))
    subgraphs = nx_subgraphs(data[i],10)
    export_graphs_to_combined_json(subgraphs, outpath="graph_json_data/multi_graphs.json", gap=18, scale=8.0)
    