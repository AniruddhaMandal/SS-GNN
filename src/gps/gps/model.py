import torch.nn as nn
from gps.models.head import ClassifierHead, LinkPredictorHead
from gps.registry import get_model
from .registry import register_model
from . import ExperimentConfig 
from . import SubgraphFeaturesBatch

@register_model('VANILLA')
def VANILLA(cfg: ExperimentConfig):
    from gps.models.vanilla import VanillaGNNClassifier 
    model = VanillaGNNClassifier(in_channels=cfg.model_config.node_feature_dim,
                            edge_dim= cfg.model_config.edge_feature_dim,
                            hidden_dim=cfg.model_config.hidden_dim,
                            out_dim=cfg.model_config.hidden_dim,
                            num_layers=cfg.model_config.mpnn_layers,
                            dropout=cfg.model_config.dropout,
                            conv_type=cfg.model_config.mpnn_type,
                            pooling=cfg.model_config.pooling)
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

class ExperimentModel(nn.Module):
    def __init__(self,
                 cfg: ExperimentConfig):
        super().__init__()
        self.is_link_prediction = (cfg.task == 'Link-Prediction')

        self.encoder = get_model(cfg.model_name)(cfg) if cfg.model_name else None
        if self.is_link_prediction:
            self.model_head =  LinkPredictorHead(in_dim=cfg.model_config.hidden_dim,
                                                 mlp_hidden=cfg.model_config.hidden_dim,
                                                 score_fn=cfg.model_config.kwargs['head_score_fn'],
                                                 dropout=cfg.model_config.dropout)
        else:
            self.model_head = ClassifierHead(in_dim=cfg.model_config.hidden_dim,
                                             hidden_dim=cfg.model_config.hidden_dim,
                                             num_classes=cfg.model_config.out_dim,
                                             dropout=cfg.model_config.dropout)
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