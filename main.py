""" Expreriment runner for subgraph sampling based graph neural networks. 
"""
import os 
import torch
import torch.nn as nn
import torch_geometric as pyg 
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import LRGBDataset
import ugs_sampler
from ss_gnn import SubgraphGINEncoder, SubgraphClassifier
import torch.optim as optim
import time

# Hyperparams
k = 15
m_per_graph = 1000
F_dim = 9
num_epochs = 60
lr = 1e-3
weight_decay = 1e-5
clip_grad_norm = 2.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset, Dataloader
train_data = LRGBDataset("data/","Peptides-func", "train")
test_data = LRGBDataset("data/","Peptides-func", "test")
val_data = LRGBDataset("data/","Peptides-func", "val")

train_loader = DataLoader(train_data,batch_size=64,shuffle=True)
test_loader = DataLoader(test_data,batch_size=64,shuffle=True)
val_loader = DataLoader(val_data,batch_size=64,shuffle=True)

# Model, optimizer, loss
encoder = SubgraphGINEncoder(in_channels=F_dim, hidden_channels=128, num_gin_layers=3)
model = SubgraphClassifier(encoder=encoder, hidden_dim=128, num_classes=10, dropout=0.1)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss()

# Optional scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

best_val_loss = float('inf')

def make_placeholders(G, m_per_graph, k, device):
    """Return placeholder sampler outputs for G graphs (all -1s / empty edges)."""
    B_total = G * m_per_graph
    nodes_t = torch.full((B_total, k), -1, dtype=torch.long)        # CPU or device
    edge_index_t = torch.empty((2, 0), dtype=torch.long)            # no edges
    edge_ptr_t = torch.zeros((B_total + 1,), dtype=torch.long)     # all zeros -> empty blocks
    graph_id_t = torch.repeat_interleave(torch.arange(G, dtype=torch.long), torch.tensor([m_per_graph]*G))
    return nodes_t, edge_index_t, edge_ptr_t, graph_id_t

# Training loop
for epoch in range(1, num_epochs + 1):
    model.train()
    epoch_loss = 0.0
    t0 = time.time()

    for step, batch in enumerate(train_loader):
        # 1) call sampler on CPU inputs (sampler expects CPU int64 tensors)
        try:
            nodes_t, edge_index_t, edge_ptr_t, graph_id_t = \
                ugs_sampler.sample_batch(batch.edge_index.cpu(), batch.ptr.cpu(), m_per_graph, k)
        except Exception as e:
            # fallback: produce placeholders for all graphs in this batch (safe)
            G = int(batch.ptr.size(0) - 1)
            nodes_t, edge_index_t, edge_ptr_t, graph_id_t = make_placeholders(G, m_per_graph, k, device)
            # optional: log once
            print(f"[warn] sampler failed; using placeholders for batch (step {step}): {e}")

        # 2) prepare model inputs
        x_global = batch.x.to(device).float()             # (N_total, F)
        targets = batch.y.to(device).float()              # (G,) long
        # nodes_t, edge_index_t, edge_ptr_t, graph_id_t can remain on CPU (model moves them inside),
        # but you may move them to device if you prefer:
        # nodes_t = nodes_t.to(device); edge_index_t = edge_index_t.to(device)
        # The encoder will .to(device) internally for these.

        # 3) forward + loss
        optimizer.zero_grad()
        logits, probs, preds, one_hot = model(x_global, nodes_t, edge_index_t, edge_ptr_t, graph_id_t, k)
        # logits: (G, C), targets: (G,)
        loss = criterion(logits, targets)
        loss.backward()

        # 4) gradient clipping + step
        if clip_grad_norm is not None and clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()

        epoch_loss += loss.item() * targets.size(0)   # accumulate sum loss (for averaging later)

        # optional logging
        if (step + 1) % 1 == 0:
            avg_loss = epoch_loss / ((step + 1) * targets.size(0))
            print(f"Epoch {epoch} Step {step+1} avg-loss {avg_loss:.4f}")

    # scheduler step
    scheduler.step()

    # epoch averages / timing
    num_train_graphs = sum([batch.ptr.size(0) - 1 for batch in train_loader])  # if loader supports len
    avg_epoch_loss = epoch_loss / (len(train_data))   # or compute from counters
    print(f"Epoch {epoch} finished in {time.time()-t0:.1f}s, avg loss: {avg_epoch_loss:.4f}")

    # -------------- validation --------------
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            try:
                nodes_t, edge_index_t, edge_ptr_t, graph_id_t = \
                    ugs_sampler.sample_batch(batch.edge_index.cpu(), batch.ptr.cpu(), m_per_graph, k)
            except Exception:
                G = int(batch.ptr.size(0) - 1)
                nodes_t, edge_index_t, edge_ptr_t, graph_id_t = make_placeholders(G, m_per_graph, k, device)

            x_global = batch.x.to(device).float()
            targets = batch.y.to(device).float()

            logits, probs, preds, one_hot = model(x_global, nodes_t, edge_index_t, edge_ptr_t, graph_id_t, k)
            loss = criterion(logits, targets)
            val_loss += loss.item() * targets.size(0)

            correct += (preds == targets.argmax()).sum().item()
            total += targets.size(0)

    val_loss = val_loss / total
    val_acc = correct / total
    print(f"Epoch {epoch} val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f}")

    # save best
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_subgraph_classifier.pth")
        print("Saved best model.")

print("Training complete.")