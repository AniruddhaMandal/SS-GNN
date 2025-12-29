import torch.nn as nn
from gps.models.head import ClassifierHead, LinkPredictorHead
from gps.models.amplified_head import build_amplified_head
from gps.registry import get_model
from .registry import register_model
from . import ExperimentConfig
from . import SubgraphFeaturesBatch

@register_model('VANILLA')
def VANILLA(cfg: ExperimentConfig):
    from gps.models.vanilla import VanillaGNNClassifier
    # For node-level tasks, skip global pooling
    pooling = 'off' if cfg.task == 'Node-Classification' else cfg.model_config.pooling
    model = VanillaGNNClassifier(in_channels=cfg.model_config.node_feature_dim,
                            edge_dim= cfg.model_config.edge_feature_dim,
                            hidden_dim=cfg.model_config.hidden_dim,
                            out_dim=cfg.model_config.hidden_dim,
                            num_layers=cfg.model_config.mpnn_layers,
                            dropout=cfg.model_config.dropout,
                            conv_type=cfg.model_config.mpnn_type,
                            pooling=pooling)
    return model

@register_model('SS-GNN')
def SSGNN(cfg: ExperimentConfig):
    from gps.models.ss_gnn import SubgraphSamplingGNNClassifier
    model = SubgraphSamplingGNNClassifier(in_channels=cfg.model_config.node_feature_dim,
                                          edge_dim=cfg.model_config.edge_feature_dim,
                                          hidden_dim=cfg.model_config.hidden_dim,
                                          num_classes=cfg.model_config.out_dim,
                                          num_layers=cfg.model_config.mpnn_layers,
                                          dropout=cfg.model_config.dropout,
                                          conv_type=cfg.model_config.mpnn_type,
                                          aggregator=cfg.model_config.pooling,
                                          temperature=cfg.model_config.temperature,
                                          pooling=cfg.model_config.subgraph_param.pooling)

    return model

@register_model('SS-GNN-WL')
def SSGNN_WL(cfg: ExperimentConfig):
    """
    SS-GNN with Frozen WL Embeddings + Learnable GNN.

    Required config parameters:
        - model_config.kwargs['wl_vocab_path']: Path to WL vocabulary file
        - model_config.kwargs['wl_dim']: WL embedding dimension (default: 64)
        - model_config.kwargs['use_node_features_in_wl']: Use node features in WL hash (default: False)
        - model_config.kwargs['wl_iterations']: Number of WL iterations (default: 3)
    """
    from gps.models.ss_gnn_wl import SubgraphSamplingGNNWithWL
    from gps.utils.wl_vocab import load_wl_vocabulary
    import os

    # Get WL vocabulary path from config
    wl_vocab_path = cfg.model_config.kwargs.get('wl_vocab_path', None)
    if wl_vocab_path is None:
        raise ValueError("SS-GNN-WL requires 'wl_vocab_path' in model_config.kwargs")

    if not os.path.exists(wl_vocab_path):
        raise FileNotFoundError(f"WL vocabulary file not found: {wl_vocab_path}")

    # Load WL vocabulary
    wl_vocab = load_wl_vocabulary(wl_vocab_path)

    # Get optional parameters
    wl_dim = cfg.model_config.kwargs.get('wl_dim', 64)
    use_node_features_in_wl = cfg.model_config.kwargs.get('use_node_features_in_wl', False)
    wl_iterations = cfg.model_config.kwargs.get('wl_iterations', 3)

    model = SubgraphSamplingGNNWithWL(
        in_channels=cfg.model_config.node_feature_dim,
        edge_dim=cfg.model_config.edge_feature_dim,
        hidden_dim=cfg.model_config.hidden_dim,
        wl_vocab=wl_vocab,
        wl_dim=wl_dim,
        num_layers=cfg.model_config.mpnn_layers,
        mlp_layers=2,
        dropout=cfg.model_config.dropout,
        conv_type=cfg.model_config.mpnn_type,
        pooling=cfg.model_config.subgraph_param.pooling,
        use_node_features_in_wl=use_node_features_in_wl,
        wl_iterations=wl_iterations
    )

    return model

class ExperimentModel(nn.Module):
    def __init__(self,
                 cfg: ExperimentConfig):
        super().__init__()
        self.is_link_prediction = (cfg.task == 'Link-Prediction')

        self.encoder = get_model(cfg.model_name)(cfg) if cfg.model_name else None

        # Determine encoder output dimension
        # For SS-GNN-WL, use combined_dim; otherwise use hidden_dim
        if hasattr(self.encoder, 'combined_dim'):
            encoder_out_dim = self.encoder.combined_dim
        else:
            encoder_out_dim = cfg.model_config.hidden_dim

        if self.is_link_prediction:
            self.model_head =  LinkPredictorHead(in_dim=encoder_out_dim,
                                                 mlp_hidden=cfg.model_config.hidden_dim,
                                                 score_fn=cfg.model_config.kwargs['head_score_fn'],
                                                 dropout=cfg.model_config.dropout)
        else:
            # Check if amplified head is requested
            head_type = cfg.model_config.kwargs.get('classifier_head_type', 'standard')
            head_scale = cfg.model_config.kwargs.get('classifier_scale', 10.0)

            if head_type == 'standard':
                self.model_head = ClassifierHead(in_dim=encoder_out_dim,
                                                 hidden_dim=cfg.model_config.hidden_dim,
                                                 num_classes=cfg.model_config.out_dim,
                                                 dropout=cfg.model_config.dropout)
            else:
                # Use amplified head
                self.model_head = build_amplified_head(
                    head_type=head_type,
                    in_dim=encoder_out_dim,
                    num_classes=cfg.model_config.out_dim,
                    hidden_dim=cfg.model_config.hidden_dim,
                    dropout=cfg.model_config.dropout,
                    scale=head_scale
                )
    def forward(self, batch: SubgraphFeaturesBatch):
        if self.is_link_prediction:
            encoding = self.encoder(batch)
            output = self.model_head(encoding, batch.edge_label_index)
        else:
            encoding = self.encoder(batch)
            output = self.model_head(encoding)
            
        return output

def build_model(cfg: ExperimentConfig)-> nn.Module:
    model = ExperimentModel(cfg)
    return model