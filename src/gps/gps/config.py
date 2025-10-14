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

    # Set Callables
    exp_config.model_fn = get_model(cfg.model_name)
    exp_config.dataloader_fn = get_dataset(cfg.dataset_name)
    exp_config.criterion_fn = get_loss(cfg.train.loss_fn) 
    exp_config.metric_fn = get_metric(cfg.train.metric)()

    # Experiment, Task, Model,Dataset Name 
    exp_config.name = cfg.name
    exp_config.task = cfg.task
    exp_config.model_name = cfg.model_name
    exp_config.dataset_name = cfg.dataset_name
    exp_config.model_config = cfg.model_config

    # set sampling parameters
    if not hasattr(exp_config.model_config,'subgraph_sampling'):
        exp_config.model_config.subgraph_sampling = False
    if exp_config.model_config.subgraph_sampling:
        assert hasattr(exp_config.model_config, "subgraph_param"), "subgraph sampling parameters missing!"
    
    # Metric and Criterion names
    exp_config.metric = cfg.train.metric
    exp_config.loss_fn = cfg.train.loss_fn

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
    exp_config.cache_dir = cfg.cache_dir
    exp_config.save_every = cfg.save_every
    exp_config.keep_last_k = cfg.keep_last_k

    # Numerical/stability
    exp_config.use_amp = cfg.use_amp
    exp_config.grad_clip =  cfg.grad_clip
    
    return exp_config
