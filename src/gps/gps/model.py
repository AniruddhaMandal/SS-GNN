from .registry import register_model
from . import ExperimentConfig

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

@register_model('SS-GNN')
def SSGNN(cfg: ExperimentConfig):
    from gps.models.ss_gnn import SubgraphSamplingGNNClassifier
    model = SubgraphSamplingGNNClassifier(in_channels=cfg.model_config.node_feature_dim,
                                          edge_dim=cfg.model_config,
                                          hidden_dim=cfg.model_config.hidden_dim,
                                          num_classes=cfg.model_config.out_dim,
                                          num_layers=cfg.model_config.mpnn_layers,
                                          dropout=cfg.model_config.dropout,
                                          conv_type=cfg.model_config.mpnn_type,
                                          graph_level_pooling=cfg.model_config.pooling,
                                          subgraph_level_pooling=cfg.model_config.subgraph_param.pooling)

    return model