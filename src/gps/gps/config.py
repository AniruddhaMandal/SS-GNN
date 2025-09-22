import os
import json
from types import SimpleNamespace
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import LRGBDataset
from .experiment import ExperimentConfig

DATASET_NAMES = ["Peptides-func"]

def load_config(f_path: str):
    if not os.path.exists(f_path):
        raise FileNotFoundError(f"`{f_path}` exists.")
    with open(f_path) as f:
        cfg = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
    return cfg

def set_config(cfg):
    exp_config = ExperimentConfig()

    # Set Callables
    exp_config.model_fn = build_model
    exp_config.dataloader_fn = build_dataloader
    exp_config.criterion_fn = build_criterion(cfg)
    exp_config.metric_fn = build_metric(cfg)

    # Experiment, Model,Dataset Name 
    exp_config.name = cfg.name
    exp_config.model_name = cfg.model_name
    exp_config.dataset_name = cfg.dataset_name
    exp_config.model_config = cfg.model_config
    
    # Metric and Criterion names
    metric = cfg.train.metric
    loss_fn = cfg.train.loss_fn

    # Training hyperparameters
    exp_config.epochs = cfg.train.epochs
    exp_config.train_batch_size = cfg.train.train_batch_size
    exp_config.val_batch_size = cfg.train.val_batch_size
    exp_config.lr = cfg.train.lr
    exp_config.weight_decay = cfg.train.weight_decay
    exp_config.optimizer = cfg.train.optimizer # 'adam' or 'adamw' or 'sgd'
    exp_config.scheduler = cfg.train.scheduler # example: {'type':'step','step_size':30,'gamma':0.1}

    # Misc
    exp_config.device = cfg.device
    exp_config.seed = cfg.seed
    exp_config.num_workers = cfg.num_workers
    exp_config.log_dir = cfg.log_dir
    exp_config.checkpoint_dir = cfg.checkpoint_dir
    exp_config.save_every = cfg.save_every
    exp_config.keep_last_k = cfg.keep_last_k

    # Numerical/stability
    exp_config.use_amp = cfg.use_amp
    exp_config.grad_clip =  cfg.grad_clip
    
    return exp_config

def build_model(cfg: ExperimentConfig):
    # return an nn.Module instance
    if cfg.model_name == "SS-GNN":
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
    
    if cfg.model_name == "VANILLA":
        from gps.models.vanilla import GINClassifier
        model = GINClassifier(in_channels=cfg.model_config.feature_dim,
                              hidden_dim=cfg.model_config.hidden_dim,
                              num_classes=cfg.model_config.out_dim,
                              dropout=cfg.model_config.dropout)
        return model

    else:
        raise ValueError(f"Unknown `model_name`:{cfg.model_name}")

def build_dataloader(cfg: ExperimentConfig):
    if(cfg.dataset_name not in DATASET_NAMES):
        raise ValueError("Unknown data set. Add to DATASET_NAMES.")

    train_data = LRGBDataset("data/",cfg.dataset_name, "train")
    test_data = LRGBDataset("data/", cfg.dataset_name, "test")
    val_data = LRGBDataset("data/", cfg.dataset_name, "val")

    train_loader = DataLoader(train_data,batch_size=cfg.train_batch_size,shuffle=True)
    test_loader = DataLoader(test_data,batch_size=cfg.val_batch_size,shuffle=True)
    val_loader = DataLoader(val_data,batch_size=cfg.val_batch_size,shuffle=True)
    return train_loader, val_loader, test_loader

def build_metric(cfg):
    """**Metric Builder**
    Returns metric function `metric_fn`.
    ```python
    def metric_fn(predict: Tensor(CPU), target: Tensor(CPU)) -> Dict
    ```
    `metric_fn` returns a dict, e.g.`{'AP': 0.345}`
    """
    if cfg.train.metric == 'AP':
        from sklearn.metrics import average_precision_score
        metric_fn = average_precision_score
        return _metric_decoretor(metric_fn, 'AP')
    else:
        raise ValueError(f"Unknown metric function: `{cfg.train.metric}`.")

def _metric_decoretor(func,name):
    def wrapper(*args, **kwargs):
        score = func(*args, **kwargs)
        return {name: score}
    return wrapper

def build_criterion(cfg):
    """**Loss/Criterion Builder**:
    Rerturns a callable `loss_fn`. 
    ```python
    def loss_fn(predict: Tesnsor, target: Tensor)-> Tensor
    ```
    """
    if cfg.train.loss_fn == "BCEWithLogitsLoss":
        from torch.nn import BCEWithLogitsLoss
        return BCEWithLogitsLoss
    else:
        raise ValueError(f"Unknown loss function. Register in CRITERION_NAMES and impliment `{cfg.train.loss_fn}`")
