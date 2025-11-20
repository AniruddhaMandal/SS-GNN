"""
Diagnostic script to understand why SS-GNN is not learning on CSL dataset.
"""
import torch
import numpy as np
from gps.experiment import Experiment
from gps.config import set_config, load_config

def analyze_model_outputs(exp, num_batches=10):
    """Analyze model outputs, gradients, and attention weights."""
    print("=" * 80)
    print("SS-GNN DIAGNOSTIC ANALYSIS")
    print("=" * 80)

    exp.model.train()
    results = {
        'losses': [],
        'logit_stds': [],
        'logit_means': [],
        'graph_emb_stds': [],
        'graph_emb_means': [],
        'attention_weights': [],
        'correct_predictions': 0,
        'total_predictions': 0,
    }

    print("\n1. FORWARD PASS ANALYSIS")
    print("-" * 80)

    for i, batch in enumerate(exp.train_loader):
        if i >= num_batches:
            break

        data, labels = exp._unpack_batch(batch)
        data = data.to(exp.device)
        labels = labels.to(exp.device)

        # Forward pass
        exp.optimizer.zero_grad()

        # Get graph embeddings (before classification head)
        graph_emb = exp.model.encoder(data)

        # Get final logits
        logits = exp.model.model_head(graph_emb)

        # Compute loss
        loss = exp.criterion(logits, labels.long())

        # Store statistics
        results['losses'].append(loss.item())
        results['logit_stds'].append(logits.std().item())
        results['logit_means'].append(logits.mean().item())
        results['graph_emb_stds'].append(graph_emb.std().item())
        results['graph_emb_means'].append(graph_emb.mean().item())

        # Check predictions
        preds = logits.argmax(dim=-1)
        results['correct_predictions'] += (preds == labels).sum().item()
        results['total_predictions'] += labels.size(0)

        # Backward pass to check gradients
        loss.backward()

        if i == 0:
            # Detailed analysis of first batch
            print(f"\nBatch {i+1} Details:")
            print(f"  Num graphs in batch: {labels.size(0)}")
            print(f"  Labels: {labels.tolist()}")
            print(f"  Predictions: {preds.tolist()}")
            print(f"  Graph embedding shape: {graph_emb.shape}")
            print(f"  Graph embedding mean: {graph_emb.mean().item():.6f}")
            print(f"  Graph embedding std: {graph_emb.std().item():.6f}")
            print(f"  Graph embedding range: [{graph_emb.min().item():.6f}, {graph_emb.max().item():.6f}]")
            print(f"  Logits shape: {logits.shape}")
            print(f"  Logits mean: {logits.mean().item():.6f}")
            print(f"  Logits std: {logits.std().item():.6f}")
            print(f"  Logits sample:\n{logits[:3].detach().cpu().numpy()}")
            print(f"  Loss: {loss.item():.6f}")

            # Check gradients
            print(f"\n  Gradient Analysis:")
            for name, param in exp.model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    param_norm = param.norm().item()
                    print(f"    {name:50s} | grad_norm: {grad_norm:10.6f} | param_norm: {param_norm:10.6f}")
                    if i == 0 and 'head' in name:  # Show more detail for head
                        print(f"      Grad mean: {param.grad.mean().item():.6f}, std: {param.grad.std().item():.6f}")

    # Summary statistics
    print("\n2. SUMMARY STATISTICS (over {} batches)".format(num_batches))
    print("-" * 80)
    print(f"  Average loss: {np.mean(results['losses']):.6f} ± {np.std(results['losses']):.6f}")
    print(f"  Average logit std: {np.mean(results['logit_stds']):.6f} ± {np.std(results['logit_stds']):.6f}")
    print(f"  Average logit mean: {np.mean(results['logit_means']):.6f} ± {np.std(results['logit_means']):.6f}")
    print(f"  Average graph_emb std: {np.mean(results['graph_emb_stds']):.6f} ± {np.std(results['graph_emb_stds']):.6f}")
    print(f"  Average graph_emb mean: {np.mean(results['graph_emb_means']):.6f} ± {np.std(results['graph_emb_means']):.6f}")
    print(f"  Training accuracy: {results['correct_predictions'] / results['total_predictions']:.4f}")

    return results

def check_subgraph_diversity(exp, num_batches=5):
    """Check if subgraph embeddings are diverse or collapsing."""
    print("\n3. SUBGRAPH EMBEDDING DIVERSITY")
    print("-" * 80)

    exp.model.eval()

    with torch.no_grad():
        for i, batch in enumerate(exp.train_loader):
            if i >= num_batches:
                break

            data, labels = exp._unpack_batch(batch)
            data = data.to(exp.device)

            # Get subgraph embeddings
            encoder = exp.model.encoder
            x_global = data.x
            edge_attr = data.edge_attr
            nodes_t = data.nodes_sampled
            edge_index_t = data.edge_index_sampled
            edge_ptr_t = data.edge_ptr
            edge_src_global_t = data.edge_src_global

            sample_emb = encoder.encode_subgraphs(
                x=x_global,
                edge_attr=edge_attr,
                nodes_t=nodes_t,
                edge_index_t=edge_index_t,
                edge_ptr_t=edge_ptr_t,
                edge_src_global_t=edge_src_global_t
            )

            if i == 0:
                print(f"\nBatch {i+1}:")
                print(f"  Num subgraphs: {sample_emb.size(0)}")
                print(f"  Subgraph embedding dim: {sample_emb.size(1)}")
                print(f"  Subgraph emb mean: {sample_emb.mean().item():.6f}")
                print(f"  Subgraph emb std: {sample_emb.std().item():.6f}")
                print(f"  Subgraph emb range: [{sample_emb.min().item():.6f}, {sample_emb.max().item():.6f}]")

                # Check if embeddings are collapsing
                pairwise_dist = torch.cdist(sample_emb, sample_emb, p=2)
                print(f"  Mean pairwise distance: {pairwise_dist.mean().item():.6f}")
                print(f"  Std pairwise distance: {pairwise_dist.std().item():.6f}")

                # Check first 5 embeddings
                print(f"  First 5 subgraph embeddings:")
                print(sample_emb[:5].cpu().numpy())

def check_attention_weights(exp, num_batches=3):
    """Check if attention aggregator is learning meaningful weights."""
    print("\n4. ATTENTION AGGREGATOR ANALYSIS")
    print("-" * 80)

    if not hasattr(exp.model.encoder.aggregator, 'attention_mlp'):
        print("  Model does not use attention aggregator. Skipping...")
        return

    exp.model.eval()

    with torch.no_grad():
        for i, batch in enumerate(exp.train_loader):
            if i >= num_batches:
                break

            data, labels = exp._unpack_batch(batch)
            data = data.to(exp.device)

            # Get subgraph embeddings
            encoder = exp.model.encoder
            x_global = data.x
            edge_attr = data.edge_attr
            nodes_t = data.nodes_sampled
            edge_index_t = data.edge_index_sampled
            edge_ptr_t = data.edge_ptr
            edge_src_global_t = data.edge_src_global
            sample_ptr_t = data.sample_ptr

            sample_emb = encoder.encode_subgraphs(
                x=x_global,
                edge_attr=edge_attr,
                nodes_t=nodes_t,
                edge_index_t=edge_index_t,
                edge_ptr_t=edge_ptr_t,
                edge_src_global_t=edge_src_global_t
            )

            # Get attention scores
            num_graphs = sample_ptr_t.size(0) - 1
            samples_per_graph = sample_ptr_t[1:] - sample_ptr_t[:-1]
            global_graph_ptr = torch.repeat_interleave(
                torch.arange(0, num_graphs, device=exp.device),
                samples_per_graph
            )

            scores = encoder.aggregator.attention_mlp(sample_emb)
            scores = scores / encoder.aggregator.temperature

            # Compute attention weights
            from torch_geometric.utils import scatter
            max_scores = scatter(scores, global_graph_ptr, dim=0, dim_size=num_graphs, reduce='max')
            max_scores = max_scores[global_graph_ptr]
            scores_exp = torch.exp(scores - max_scores)
            scores_sum = scatter(scores_exp, global_graph_ptr, dim=0, dim_size=num_graphs, reduce='sum')
            scores_sum = scores_sum[global_graph_ptr]
            attention_weights = scores_exp / (scores_sum + 1e-8)

            if i == 0:
                print(f"\nBatch {i+1}:")
                print(f"  Num graphs: {num_graphs}")
                print(f"  Samples per graph: {samples_per_graph.tolist()}")
                print(f"  Attention weights mean: {attention_weights.mean().item():.6f}")
                print(f"  Attention weights std: {attention_weights.std().item():.6f}")
                print(f"  Attention weights range: [{attention_weights.min().item():.6f}, {attention_weights.max().item():.6f}]")
                print(f"  Attention weights for first graph: {attention_weights[:samples_per_graph[0].item()].squeeze().tolist()}")
                print(f"  Are weights uniform? (std < 0.01): {attention_weights.std().item() < 0.01}")

def main():
    # Load config
    config_file_path = 'configs/ss_gnn/SYNTHETIC/CSL/gin-k4.json'
    dict_cfg = load_config(config_file_path)
    dict_cfg['tracker'] = 'Off'
    dict_cfg['device'] = 'cpu'
    cfg = set_config(dict_cfg)

    # Build experiment
    print("Building experiment...")
    exp = Experiment(cfg=cfg)
    print("Experiment built successfully!\n")

    # Run diagnostics
    analyze_model_outputs(exp, num_batches=10)
    check_subgraph_diversity(exp, num_batches=5)
    check_attention_weights(exp, num_batches=3)

    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
