from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ------- Configuration dataclass -------

@dataclass
class ExperimentConfig:
    # Callables
    model_fn: Optional[Callable[["ExperimentConfig"], nn.Module]] = None
    dataloader_fn: Optional[Callable[["ExperimentConfig"], Tuple[DataLoader, DataLoader, Optional[DataLoader]]]] = None
    criterion_fn: Optional[Callable[["ExperimentConfig"], nn.Module]] = None
    metric_fn: Optional[Callable[[torch.Tensor, torch.Tensor], Dict[str, float]]] = None

    # Model / data identifiers (for logging filenames)
    name: str = "exp"
    task: Optional[str] = None
    dataset_name: Optional[str] = None
    model_name: Optional[str] = None
    model_config: Optional[SimpleNamespace] = None 

    # Metric and Criterion names
    metric: Optional[str] = None
    loss_fn: Optional[str] = None

    # Training hyperparameters
    epochs: int = 100
    train_batch_size: int = 32
    val_batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    optimizer: str = "adam"  # 'adam' or 'adamw' or 'sgd'
    scheduler: Optional[Dict[str, Any]] = None  # example: {'type':'step','step_size':30,'gamma':0.1}

    # Misc
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    num_workers: int = 4
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    save_every: int = 1
    keep_last_k: int = 3

    # Numerical/stability
    use_amp: bool = False
    grad_clip: Optional[float] = None  # clip norm value

    # Extra container for user-specific kwargs
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    dataloader_kwargs: Dict[str, Any] = field(default_factory=dict)
    criterion_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Resume
    resume_from: Optional[str] = None
