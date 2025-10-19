from .registry import register_model
from . import ExperimentConfig

@register_model('SS-GNN')
def SSGNN(cfg: ExperimentConfig):
    from ss_gnn import SubgraphClassifier, SubgraphGINEncoder, SubgraphTransformerAggregator
    encoder = SubgraphGINEncoder(
        in_channels=cfg.model_config.feature_dim,
        hidden_channels=cfg.model_config.hidden_dim,
        num_gin_layers=cfg.model_config.mpnn_layers,
        dropout=cfg.model_config.dropout)
    aggar = SubgraphTransformerAggregator(encoder=encoder,
                                            hidden_dim=cfg.model_config.hidden_dim,
                                            n_heads=cfg.model_config.transformer_heads,
                                            dim_feedforward=cfg.model_config.transformer_dim,
                                            dropout=cfg.model_config.dropout)
    model = SubgraphClassifier(encoder=aggar,
                                hidden_dim=cfg.model_config.hidden_dim,
                                num_classes=cfg.model_config.out_dim,
                                dropout=cfg.model_config.dropout)
    return model

@register_model('VANILLA')
def VANILLA(cfg: ExperimentConfig):
    from gps.models.vanilla import VanillaGNNClassifier 
    model = VanillaGNNClassifier(in_channels=cfg.model_config.node_feature_dim,
                            edge_dim= cfg.model_config.edge_feature_dim,
                            hidden_dim=cfg.model_config.hidden_dim,
                            num_classes=cfg.model_config.out_dim,
                            num_layers=cfg.model_config.mpnn_layers,
                            dropout=cfg.model_config.dropout,
                            conv_type=cfg.model_config.mpnn_type,
                            pooling=cfg.model_config.pooling)
    return model
