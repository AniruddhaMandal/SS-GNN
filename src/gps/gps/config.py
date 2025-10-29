import os
import json
from typing import Dict, Any
from gps.experiment import ExperimentConfig
from . import ExperimentConfig, merge_into_dataclass
from .registry import get_model, get_dataset, get_metric, get_loss
from . import model
from . import datasets
from . import loss
from . import metric

def load_config(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"`{path}` doesn't exist.")
    with open(path) as f:
        return json.load(f)

def set_config(cfg_dict: dict,
               *,
               strict: bool = True) -> ExperimentConfig:
    """
    Build a fully-populated ExperimentConfig:
      - merge cfg_dict onto dataclass defaults
      - resolve factory functions into callables
      - validate (optional)
    """
    exp = ExperimentConfig()                 # defaults
    merge_into_dataclass(exp, cfg_dict)      # overlay file values

    # --- Resolve callables based on names present in exp ---
    exp.model_fn = get_model(exp.model_name) if exp.model_name else None
    exp.dataloader_fn = get_dataset(exp.dataset_name) if exp.dataset_name else None
    exp.criterion_fn = get_loss(exp.train.loss_fn) if exp.train.loss_fn else None
    exp.metric_fn = get_metric(exp.train.metric)() if exp.train.metric else None

    # --- Optional validation (fail fast with helpful hints) ---
    missing = []
    if exp.model_fn is None:
        missing.append("model_fn (set `model_name`)")
    if exp.dataloader_fn is None:
        missing.append("dataloader_fn (set `dataset_name`)")
    if exp.criterion_fn is None:
        missing.append("criterion_fn (set `train.loss_fn`)")

    if strict and missing:
        bullet = "\n  - ".join(missing)
        raise ValueError(f"Incomplete configuration. Please provide:\n  - {bullet}")

    return exp