# utils/split_and_loader.py
import os, json, random
from typing import Dict, Any, Tuple
import numpy as np
import torch
from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split

from .. import ExperimentConfig

def _ensure_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def _save_splits(path: str, splits: Dict[str, list]):
    _ensure_dir(path)
    # convert to plain lists of ints
    serial = {k: [int(i) for i in v] for k, v in splits.items()}
    with open(path, "w") as f:
        json.dump(serial, f)

def _load_splits(path: str):
    with open(path) as f:
        j = json.load(f)
    return {k: [int(i) for i in v] for k, v in j.items()}

def dataset_has_masks(dataset) -> bool:
    try:
        d0 = dataset[0]
        return any(hasattr(d0, m) for m in ("train_mask", "val_mask", "test_mask"))
    except Exception:
        return False

def build_or_load_splits(dataset, cfg: ExperimentConfig):
    """
    Returns dict with keys 'train','val','test' -> lists of indices.
    Detection order:
      1) OGB-style get_idx_split()
      2) data-level masks (train_mask/val_mask/test_mask) on dataset[0]
      3) cached splits file (cfg['split_cache_dir'])
      4) deterministic stratified random split (and cache)
    """
    cache_dir = cfg.cache_dir
    cache_path = os.path.join(cache_dir, f"{cfg.dataset_name}_splits.json")
    seed = int(cfg.seed)

    # 1) OGB-style
    if hasattr(dataset, "get_idx_split"):
        splits = dataset.get_idx_split()  # returns dict {'train': [...], ...}
        # ensure lists of ints
        return {k: [int(i) for i in v] for k, v in splits.items()}

    # 2) mask-based (per-graph or global)
    if dataset_has_masks(dataset):
        train_idx, val_idx, test_idx = [], [], []
        for i, d in enumerate(dataset):
            if getattr(d, "train_mask", None) is not None and getattr(d, "train_mask").item():
                train_idx.append(i)
            elif getattr(d, "val_mask", None) is not None and getattr(d, "val_mask").item():
                val_idx.append(i)
            elif getattr(d, "test_mask", None) is not None and getattr(d, "test_mask").item():
                test_idx.append(i)
        # if masks are boolean tensors per-node/graph, convert accordingly above
        if len(train_idx) + len(val_idx) + len(test_idx) > 0:
            return {"train": train_idx, "val": val_idx, "test": test_idx}

    # 3) cached splits
    if os.path.exists(cache_path):
        return _load_splits(cache_path)

    # 4) deterministic stratified split and cache
    n = len(dataset)
    idx = list(range(n))
    # try to build labels for stratification; fallback to None
    try:
        labels = []
        for d in dataset:
            y = getattr(d, "y", None)
            # handle tensors / scalars / one-hot
            if y is None:
                labels.append(None)
            else:
                y_ = int(np.asarray(y).argmax()) if (getattr(y, "ndim", 0) and np.asarray(y).ndim > 0) else int(np.asarray(y))
                labels.append(y_)
        strat = labels if any(lbl is not None for lbl in labels) else None
    except Exception:
        strat = None

    # first split train vs rest
    try: 
        train_ratio = cfg.train.train_ratio
        val_ratio = cfg.train.val_ratio
    except:
        train_ratio = 0.8
        val_ratio = 0.1
    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio <= 0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")

    if strat is not None:
        # sklearn expects no None in stratify; build list for stratify param
        strat_list = np.array([s if s is not None else -1 for s in strat])
        # remove -1 rows if all -1 -> treat as no stratify
        if np.all(strat_list == -1):
            strat_list = None
        else:
            strat_list = strat_list
    else:
        strat_list = None

    # deterministic seed
    rng = np.random.RandomState(seed)
    if strat_list is not None:
        train_idx, rest = train_test_split(idx, train_size=train_ratio, stratify=strat_list, random_state=seed)
        if len(rest) == 0:
            val_idx, test_idx = [], []
        else:
            # compute val fraction relative to rest
            val_frac = val_ratio / (1.0 - train_ratio)
            rest_strat = [strat_list[i] for i in rest] if strat_list is not None else None
            val_idx, test_idx = train_test_split(rest, train_size=val_frac, stratify=rest_strat, random_state=seed)
    else:
        # no stratify
        rng.shuffle(idx)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train + n_val]
        test_idx = idx[n_train + n_val:]

    splits = {"train": train_idx, "val": val_idx, "test": test_idx}
    _save_splits(cache_path, splits)
    return splits

from collections.abc import Mapping, Sequence

def _is_split_container(obj):
    # dict-like: {"train": ds, "val": ds, "test": ds}
    if isinstance(obj, Mapping):
        return all(k in obj for k in ("train","val","test"))
    # tuple/list: (train_ds, val_ds, test_ds)
    if isinstance(obj, Sequence) and len(obj) == 3:
        return True
    # object with attributes .train .val .test
    if all(hasattr(obj, a) for a in ("train","val","test")):
        return True
    return False

def build_dataloaders_from_dataset(dataset_or_splits, cfg: ExperimentConfig, collate_fn=None, pin_memory=False):
    """
    Accepts either:
      - a single Dataset (will create/cache splits), OR
      - a split-container: dict('train','val','test') or (train_ds,val_ds,test_ds)
      - an object implementing get_idx_split() (OGB-style) or having train/val/test attrs.
    """
    # 1) If user passed a container with splits, extract them verbatim
    if _is_split_container(dataset_or_splits):
        if isinstance(dataset_or_splits, Mapping):
            train_ds = dataset_or_splits["train"]
            val_ds   = dataset_or_splits["val"]
            test_ds  = dataset_or_splits["test"]
        elif isinstance(dataset_or_splits, Sequence):
            train_ds, val_ds, test_ds = dataset_or_splits
        else:
            train_ds, val_ds, test_ds = dataset_or_splits.train, dataset_or_splits.val, dataset_or_splits.test

    else:
        # 2) Single dataset object: fall back to automatic split builder (existing code)
        dataset = dataset_or_splits
        # if dataset exposes get_idx_split (OGB/LRGB-style), use it
        if hasattr(dataset, "get_idx_split"):
            splits = dataset.get_idx_split()
            # splits may be dict of numpy arrays/tensors -> make Subset wrappers
            train_ds = torch.utils.data.Subset(dataset, list(splits["train"]))
            val_ds   = torch.utils.data.Subset(dataset, list(splits["valid"] if "valid" in splits else splits.get("val")))
            test_ds  = torch.utils.data.Subset(dataset, list(splits["test"]))
        else:
            # fallback: your deterministic stratified split + cache (reuse earlier function)
            splits = build_or_load_splits(dataset, cfg)
            train_ds = torch.utils.data.Subset(dataset, splits["train"])
            val_ds   = torch.utils.data.Subset(dataset, splits["val"])
            test_ds  = torch.utils.data.Subset(dataset, splits["test"])

    # 3) Build DataLoaders (same as before)
    train_loader = DataLoader(train_ds, batch_size=int(cfg.train_batch_size), shuffle=False,
                                num_workers=int(cfg.num_workers), collate_fn=collate_fn, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=int(cfg.val_batch_size), shuffle=False, 
                                num_workers=cfg.num_workers//2, collate_fn=collate_fn, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=cfg.val_batch_size, shuffle=False, 
                                num_workers=cfg.num_workers//2, collate_fn=collate_fn, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader
