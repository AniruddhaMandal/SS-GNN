from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Literal, Mapping

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ----- Type aliases -----
PoolingType = Literal["mean", "max", "add"]
OptimizerType = Literal["adam", "adamw", "sgd"]
MpnnType = Literal["gcn", "gin", "graphsage"]

ModelFactory = Callable[["ExperimentConfig"], nn.Module]
DataloaderFactory = Callable[["ExperimentConfig"], Tuple[DataLoader, DataLoader, Optional[DataLoader]]]
CriterionFactory = Callable[["ExperimentConfig"], nn.Module]
MetricFn = Callable[[torch.Tensor, torch.Tensor], Dict[str, float]]


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# ----- Nested/section dataclasses -----
@dataclass
class SubgraphParam:
    k: int = 5
    m: int = 10
    pooling: Optional[PoolingType] = "mean"

@dataclass
class SchedulerCfg:
    type: Optional[str] = None
    setp_size: Optional[int] = None
    gamma: Optional[float] = None
    patience: Optional[int] = None

@dataclass
class ModelConfig:
    node_feature_dim: Optional[int] = None
    edge_feature_dim: Optional[int] = None
    hidden_dim: Optional[int] = None
    out_dim: Optional[int] = None
    mpnn_layers: Optional[int] = None
    dropout: Optional[float] = None
    pooling: Optional[PoolingType] = None
    subgraph_sampling: bool = False
    subgraph_param: SubgraphParam = field(default_factory=SubgraphParam)
    mpnn_type: MpnnType = "gcn"
    # Extra user-defined knobs
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainConfig:
    # Training hyperparameters
    epochs: int = 100
    train_batch_size: int = 32
    val_batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    optimizer: OptimizerType = "adam"  # "adam" | "adamw" | "sgd"
    scheduler: Optional[SchedulerCfg] = field(default_factory=SchedulerCfg)  

    # Objective / metrics
    metric: Optional[str] = None
    loss_fn: Optional[str] = None

    # Numerical / stability
    use_amp: bool = False
    grad_clip: Optional[float] = None  # clip norm value

    # Extra kwargs per block
    dataloader_kwargs: Dict[str, Any] = field(default_factory=dict)
    criterion_kwargs: Dict[str, Any] = field(default_factory=dict)


# ----- Top-level experiment config -----
@dataclass
class ExperimentConfig:
    # Identifiers
    name: str = "exp"
    task: Optional[str] = None
    dataset_name: Optional[str] = None
    model_name: Optional[str] = None

    # Structured sections
    model_config: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # Misc/runtime
    device: str = field(default_factory=_default_device)
    seed: int = 42
    num_workers: int = 4
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    save_every: int = 1
    keep_last_k: int = 3
    cache_dir: Optional[str] = None
    resume_from: Optional[str] = None

    # Derived/callables (set after parsing)
    model_fn: Optional[ModelFactory] = None
    dataloader_fn: Optional[DataloaderFactory] = None
    criterion_fn: Optional[CriterionFactory] = None
    metric_fn: Optional[MetricFn] = None


# ----- Minimal recursive merge helper -----
def merge_into_dataclass(dc, src: Mapping[str, Any] | None):
    """
    Recursively copy values from dict-like `src` into dataclass `dc`,
    preserving defaults for any keys not present in `src`.
    """
    if src is None:
        return dc
    if not isinstance(src, Mapping):
        raise TypeError("merge_into_dataclass expects a Mapping as `src`")

    for f in fields(dc):
        if f.name not in src or src[f.name] is None:
            continue

        incoming = src[f.name]
        current = getattr(dc, f.name)

        if is_dataclass(current) and isinstance(incoming, Mapping):
            merge_into_dataclass(current, incoming)  # mutate nested dc
        else:
            setattr(dc, f.name, incoming)

    return dc
