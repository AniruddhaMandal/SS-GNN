import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter
from .registry import register_aggregator

@register_aggregator('attention')
class AttentionAggregator(nn.Module):
    def __init__(self, hidden_dim, temperature=2.0):
        super().__init__()
        self.temperature = temperature
        self.attention_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, subgraph_embeddings, batch):
        """
        Args:
            subgraph_embeddings: [total_subgraphs, hidden_dim]
            batch: [total_subgraphs] - which graph each subgraph belongs to
        Returns:
            graph_embeddings: [num_graphs, hidden_dim]
        """
        num_graphs = batch.max().item() + 1
        scores = self.attention_mlp(subgraph_embeddings)  # [total_subgraphs, 1]
        scores = scores / self.temperature

        max_scores = scatter(scores, batch, dim=0, dim_size=num_graphs, reduce='max')  # [num_graphs, 1]
        max_scores = max_scores[batch]  # Broadcast back
        scores_exp = torch.exp(scores - max_scores)
        scores_sum = scatter(scores_exp, batch, dim=0, dim_size=num_graphs, reduce='sum')  # [num_graphs, 1]
        scores_sum = scores_sum[batch] 
        attention_weights = scores_exp / (scores_sum + 1e-8)  # [total_subgraphs, 1]
        
        weighted_embeddings = attention_weights * subgraph_embeddings  # [total_subgraphs, hidden_dim]
        graph_embeddings = scatter(weighted_embeddings, batch, dim=0, dim_size=num_graphs, reduce='sum')
        
        return graph_embeddings