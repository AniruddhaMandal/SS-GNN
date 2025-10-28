import os
import json
from types import SimpleNamespace
from gps.experiment import ExperimentConfig
from .registry import get_model, get_dataset, get_metric, get_loss
from . import model
from . import datasets
from . import loss
from . import metric

def load_config(f_path: str):
    if not os.path.exists(f_path):
        raise FileNotFoundError(f"`{f_path}` doesn't exists.")
    with open(f_path) as f:
        cfg = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
    return cfg

def set_config(cfg):
    exp_config = ExperimentConfig()

    # --- Set Callables ---
    exp_config.model_fn = get_model(getattr(cfg, "model_name", None))
    exp_config.dataloader_fn = get_dataset(getattr(cfg, "dataset_name", None))
    exp_config.criterion_fn = get_loss(getattr(getattr(cfg, "train", {}), "loss_fn", None))
    exp_config.metric_fn = (
        get_metric(getattr(getattr(cfg, "train", {}), "metric", None))()
        if getattr(getattr(cfg, "train", {}), "metric", None) is not None
        else None
    )

    # --- Experiment, Task, Model, Dataset ---
    exp_config.name = getattr(cfg, "name", None)
    exp_config.task = getattr(cfg, "task", None)
    exp_config.model_name = getattr(cfg, "model_name", None)
    exp_config.dataset_name = getattr(cfg, "dataset_name", None)
    exp_config.model_config = getattr(cfg, "model_config", None)

    # --- Sampling parameters ---
    if exp_config.model_config is not None:
        if not hasattr(exp_config.model_config, "subgraph_sampling"):
            exp_config.model_config.subgraph_sampling = False

        if exp_config.model_config.subgraph_sampling:
            assert hasattr(exp_config.model_config, "subgraph_param"), "subgraph sampling parameters missing!"

        if not hasattr(exp_config.model_config, "edge_feature_dim"):
            exp_config.model_config.edge_feature_dim = None

        if not hasattr(exp_config.model_config, "mpnn_type"):
            exp_config.model_config.mpnn_type = "gcn"

    # --- Metric and Criterion names ---
    exp_config.metric = getattr(getattr(cfg, "train", {}), "metric", None)
    exp_config.loss_fn = getattr(getattr(cfg, "train", {}), "loss_fn", None)

    # --- Training hyperparameters ---
    exp_config.train = getattr(cfg, "train", None)
    exp_config.epochs = getattr(getattr(cfg, "train", {}), "epochs", None)
    exp_config.train_batch_size = getattr(getattr(cfg, "train", {}), "train_batch_size", None)
    exp_config.val_batch_size = getattr(getattr(cfg, "train", {}), "val_batch_size", None)
    exp_config.lr = getattr(getattr(cfg, "train", {}), "lr", None)
    exp_config.weight_decay = getattr(getattr(cfg, "train", {}), "weight_decay", None)
    exp_config.optimizer = getattr(getattr(cfg, "train", {}), "optimizer", None)
    exp_config.scheduler = getattr(getattr(cfg, "train", {}), "scheduler", None)

    # --- Misc ---
    exp_config.device = getattr(cfg, "device", None)
    exp_config.seed = getattr(cfg, "seed", None)
    exp_config.num_workers = getattr(cfg, "num_workers", None)
    exp_config.log_dir = getattr(cfg, "log_dir", None)
    exp_config.checkpoint_dir = getattr(cfg, "checkpoint_dir", None)
    exp_config.cache_dir = getattr(cfg, "cache_dir", None)
    exp_config.save_every = getattr(cfg, "save_every", None)
    exp_config.keep_last_k = getattr(cfg, "keep_last_k", None)

    # --- Numerical/stability ---
    exp_config.use_amp = getattr(cfg, "use_amp", None)
    exp_config.grad_clip = getattr(cfg, "grad_clip", None)

    return exp_config
