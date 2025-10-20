"""
Experiment runner template
==========================

A flexible Experiment class and ExperimentConfig dataclass you can drop into your project.

Purpose
-------
- Provide structure for running training/evaluation experiments for PyTorch models (including
  GNNs with PyTorch Geometric) with minimal changes.
- The class is intentionally generic: you supply small callables or override hooks to instantiate
  datasets / models / metrics. This file provides the scaffolding and logging/checkpointing.

How to use
----------
- Fill `ExperimentConfig` fields or construct one and pass callables in `model_fn`, `dataloader_fn`.
- `model_fn(cfg)` should return an nn.Module (uninitialized weights OK).
- `dataloader_fn(cfg)` should return a tuple (train_loader, val_loader, test_loader) or
  `(train_loader, val_loader)` (test optional).
- Optionally provide `criterion_fn`, `metric_fn` or override `evaluate`.

Notes
-----
- This template supports AMP (torch.cuda.amp), gradient clipping, LR scheduler hooks, checkpointing,
  TensorBoard logging, deterministic seeding, and simple CLI friendly prints.
- You can subclass `Experiment` and override hooks for complex dataset/model construction.

"""

from __future__ import annotations
import os
import time
import json
import shutil
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Literal 
import random 
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import ugs_sampler
from tqdm.auto import tqdm
from . import ExperimentConfig

# ------- Experiment class -------

class Experiment:
    """**Experiment scaffolding. Override or pass callables via ExperimentConfig.**

    Key hooks to override or supply in config:
      - `model_fn(cfg)` -> `nn.Module`
      - `dataloader_fn(cfg)` -> (train_loader, val_loader, test_loader)
      - `criterion_fn(cfg)` -> loss module
      - `metric_fn(preds, targets)` -> `dict` of metrics

    The default training/eval code is generic for classification/regression tasks. For
    more custom tasks (structured outputs, GNNs with special batching), override
    `train_one_epoch` and `evaluate`.

    ## Example usage snippet 
    (fill these callables)

    ```python
    def my_model_fn(cfg):
        # return an nn.Module instance
        return MyModel(**cfg.model_kwargs)

    def my_dataloader_fn(cfg):
        # return train_loader, val_loader, test_loader (test optional)
        # e.g. using torch_geometric's DataLoader or standard DataLoader
        return train_loader, val_loader, test_loader

    def my_metric_fn(preds, targets):
        # preds / targets are cpu tensors. Return dict of metrics, e.g. {'accuracy':acc}
        from sklearn.metrics import accuracy_score
        acc = accuracy_score(targets.numpy(), preds.numpy())
        return {'accuracy': acc, 'metric': acc}

    cfg = ExperimentConfig(name='myexp', model_fn=my_model_fn, dataloader_fn=my_dataloader_fn,
                           criterion_fn=lambda c: nn.CrossEntropyLoss(), metric_fn=my_metric_fn)
    exp = Experiment(cfg)
    exp.train()
    ```

    Customize by subclassing Experiment and overriding `train_one_epoch` and `evaluate` for advanced flows.
    """

    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        os.makedirs(cfg.log_dir, exist_ok=True)
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)

        # logging
        self.logger = self._setup_logger()
        self.writer = SummaryWriter(log_dir=os.path.join(cfg.log_dir, cfg.name))

        # deterministic
        self._set_seed(cfg.seed)

        # placeholders (filled in build)
        self.model: Optional[nn.Module] = None
        self.criterion: Optional[nn.Module] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler = None
        self.scaler: Optional[torch.cuda.amp.GradScaler] = None

        # data loaders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        # bookkeeping
        self.down_metrics = ['MAE']
        if self.cfg.metric in self.down_metrics:
            self.best_metric = float('inf')
        else:
            self.best_metric = -float('inf')
        self.history = {"train_loss": [], "val_loss": []}

        # Build components
        self.build()

    # ---------- Setup helpers ----------
    def _setup_logger(self):
        logger = logging.getLogger(self.cfg.name)
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        ch.setFormatter(fmt)
        if not logger.handlers:
            logger.addHandler(ch)
        return logger

    def _set_seed(self, seed: int):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # ---------- Build components (override or supply fns) ----------
    def build(self):
        self.logger.info("Building experiment components...")
        # model
        if callable(self.cfg.model_fn):
            self.model = self.cfg.model_fn(self.cfg)
        else:
            raise NotImplementedError("Provide cfg.model_fn(cfg) returning an nn.Module")
        self.model.to(self.device)

        # criterion
        if callable(self.cfg.criterion_fn):
            self.criterion = self.cfg.criterion_fn()
        else:
            # default: cross entropy
            self.criterion = nn.CrossEntropyLoss(**self.cfg.criterion_kwargs)

        # dataloaders
        if callable(self.cfg.dataloader_fn):
            loaders = self.cfg.dataloader_fn(self.cfg)
            if isinstance(loaders, tuple) and (2 <= len(loaders) <= 3):
                self.train_loader, self.val_loader = loaders[0], loaders[1]
                self.test_loader = loaders[2] if len(loaders) == 3 else None
            else:
                raise ValueError("dataloader_fn must return (train_loader, val_loader, [test_loader])")
        else:
            raise NotImplementedError("Provide cfg.dataloader_fn(cfg) returning dataloaders")

        # optimizer
        self.optimizer = self._build_optimizer()

        # scheduler (optional)
        self.scheduler = self._build_scheduler()

        # AMP scaler
        if self.cfg.use_amp:
            self.scaler = torch.amp.GradScaler()

        # optionally resume
        if self.cfg.resume_from:
            self.load_checkpoint(self.cfg.resume_from)

        self.logger.info("Build complete: device=%s, model=%s", self.device, type(self.model).__name__)

    def _build_optimizer(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        if self.cfg.optimizer.lower() in ('adam', 'adamw'):
            opt_cls = optim.AdamW if self.cfg.optimizer.lower() == 'adamw' else optim.Adam
            return opt_cls(params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        elif self.cfg.optimizer.lower() == 'sgd':
            return optim.SGD(params, lr=self.cfg.lr, momentum=0.9, weight_decay=self.cfg.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {self.cfg.optimizer}")

    def _build_scheduler(self):
        sch = None
        s = self.cfg.scheduler
        if not s:
            return None
        if s.type == 'step':
            sch = optim.lr_scheduler.StepLR(self.optimizer, step_size=s.step_size, gamma=s.gamma)
        elif s.type == 'cosine':
            sch = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.cfg.epochs)
        elif s.type == 'reduce_on_plateau':
            sch = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=s.patience)
        else:
            self.logger.warning("Unknown scheduler type: %s", s.type)
        return sch

    def _initial_fault_check(self):
        _batch = next(iter(self.train_loader))
        _feat_dim = _batch.x.size(-1)
        if hasattr(self.cfg.model_config, 'node_feature_dim'):
            _model_in_dim = self.cfg.model_config.node_feature_dim
        if(_model_in_dim != _feat_dim):
            raise ValueError(f"model input dim({_model_in_dim}) not equal to node feature dim({_feat_dim}).")

    # ---------- Training / evaluation ----------
    def train(self):
        self.logger.info("Starting training for %d epochs", self.cfg.epochs)
        #self._initial_fault_check()
        for epoch in range(1, self.cfg.epochs + 1):
            t0 = time.time()
            train_stats = self.train_one_epoch(epoch)
            val_stats = self.evaluate(epoch)

            # logging
            self.history['train_loss'].append(train_stats.get('loss', None))
            self.history['val_loss'].append(val_stats.get('loss', None))

            # scheduler step (handle reduce_on_plateau separately)
            if self.scheduler is not None and not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step()
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_stats.get('loss'))

            # save checkpoint
            if epoch % self.cfg.save_every == 0:
                metric_for_save = val_stats.get("metric")
                self.save_checkpoint(epoch, metric_for_save)

            dt = time.time() - t0
            self.logger.info("Epoch %03d | time %.1fs | train_loss %.4f | val_loss %.4f | val_metric %.5f\n",
                              epoch, dt, train_stats.get('loss', float('nan')), val_stats.get('loss', float('nan')),
                              float(val_stats.get('metric', None)))

        fname_best_model = f"best_model.pth"
        path_best_model = os.path.join(self.cfg.checkpoint_dir, fname_best_model)
        self.load_checkpoint(path_best_model)
        test_best_stats = self.evaluate(split='test')
        train_best_stats = self.evaluate(split='train')
        self.logger.info("Best model metric \n\tTest data: %s \n\tTrain data: %s \n\tVal data: %s", test_best_stats.get('metric', None), train_best_stats.get('metric', None),self.best_metric)
        self.save_result(test_best_stats.get('metric', None), train_best_stats.get('metric', None),self.best_metric)

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """Default training loop. Override for custom tasks.

        Returns a dict of statistics (must contain 'loss').
        """
        self.model.train()
        running_loss = 0.0
        n_examples = 0

        pbar = tqdm(self.train_loader, desc=f"Train {epoch}")
        for batch in pbar:
            # Expectation: user dataloader yields a tuple (inputs..., labels)
            # You can adapt these lines to your use-case. For GNNs using torch_geometric,
            # you might receive a Batch object where `batch.x, batch.edge_index, batch.batch`
            # If subgraph sampling is required then return: 
            # `x_global`, `nodes_t`, `edge_index_t`, `edge_ptr_t`, `graph_id_t`, `k`
            inputs, labels = self._unpack_batch(batch, self.cfg.model_config.subgraph_sampling)
            inputs = self._to_device(inputs)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=self.cfg.use_amp):
                logits = self.model(*inputs) 
                #user-provided criterion must accept (outputs, labels)
                if self.cfg.task == "Binary-Classification":
                    loss = self.criterion(logits, labels.long())
                if self.cfg.task == "Multi-Lable-Binary-Classification":
                    loss = self.criterion(logits, labels.float())
                if self.cfg.task == "Multi-Target-Regression":
                    loss = self.criterion(logits, labels.float())
                if self.cfg.task == "Multi-Class-Classification":
                    loss = self.criterion(logits, labels.long())

            # backward
            if self.cfg.use_amp:
                assert self.scaler is not None
                self.scaler.scale(loss).backward()
                if self.cfg.grad_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.cfg.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.optimizer.step()

            # bookkeeping
            batch_size = self._get_batch_size(batch)
            running_loss += loss.item() * batch_size
            n_examples += batch_size
            avg_loss = running_loss / max(1, n_examples)
            pbar.set_postfix(loss=avg_loss)

        return {"loss": avg_loss}

    def evaluate(self, epoch: Optional[int] = None, split: Optional[Literal['test', 'train', 'val']] = 'val') -> Dict[str, Any]:
        """Evaluate on validation set. Returns dict with 'loss' and optionally 'metric'."""
        self.model.eval()
        running_loss = 0.0
        n_examples = 0
        all_logits = []
        all_targets = []
        loaders = {
            "test": self.test_loader,
            "train": self.train_loader,
            "val": self.val_loader,
        }

        if split not in loaders:
            raise ValueError(f"Unknown split type ({split}) in evaluate.")

        loader = loaders[split]

        pbar = tqdm(loader, desc=f"{split} {epoch}" if epoch else f"{split}")
        with torch.no_grad():
            for batch in pbar:
                inputs, labels = self._unpack_batch(batch, self.cfg.model_config.subgraph_sampling)
                inputs = self._to_device(inputs)
                labels = labels.to(self.device)
                with torch.amp.autocast('cuda', enabled=self.cfg.use_amp):
                    logits = self.model(*inputs) 
                    #user-provided criterion must accept (outputs, labels)
                    if self.cfg.task == "Binary-Classification":
                        loss = self.criterion(logits, labels.long())
                    if self.cfg.task == "Multi-Lable-Binary-Classification":
                        loss = self.criterion(logits, labels.float())
                    if self.cfg.task == "Multi-Target-Regression":
                        loss = self.criterion(logits, labels.float())
                    if self.cfg.task == "Multi-Class-Classification":
                        loss = self.criterion(logits, labels.long())

                # collect logits and targets
                all_logits.append(logits.detach().cpu())
                all_targets.append(labels.detach().cpu())

                batch_size = self._get_batch_size(batch)
                running_loss += loss.item() * batch_size
                n_examples += batch_size

        avg_loss = running_loss / max(1, n_examples)
        all_logits = torch.cat(all_logits, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        metrics = {}
        if callable(self.cfg.metric_fn):
            try:
                # IMPORTANT: average_precision_score expects (y_true, y_score)
                # y_true shape: (N,) for binary or (N, n_classes) for multilabel
                # y_score shape: same as y_true for multilabel
                if self.cfg.task == "Multi-Lable-Binary-Classification":
                    all_probs = torch.sigmoid(all_logits)
                    all_preds = (all_probs > 0.5).long()      # [B, num_labels]
                    metrics = self.cfg.metric_fn(all_preds.numpy(), all_targets.numpy())
                if self.cfg.task == "Binary-Classification":
                    all_preds = torch.argmax(all_logits, dim=1)   # [B]
                    metrics = self.cfg.metric_fn(all_preds.numpy(), all_targets.numpy())
                if self.cfg.task == "Multi-Target-Regression":
                    metrics = self.cfg.metric_fn(all_logits.numpy(), all_targets.numpy())
                if self.cfg.task == "Multi-Class-Classification":
                    all_probs  = all_logits.softmax(dim=-1)
                    all_preds  = all_probs.argmax(dim=-1)
                    metrics = self.cfg.metric_fn(all_preds.numpy(), all_targets.numpy())
                
            except Exception as e:
                self.logger.warning("metric_fn failed: %s", e)
        print("Metrics: ",metrics)
        metric_val = metrics.get(self.cfg.metric)

        # tensorboard
        if epoch is not None:
            self.writer.add_scalar('val/loss', avg_loss, epoch)
            if metric_val is not None:
                self.writer.add_scalar('val/metric', metric_val, epoch)

        return {"loss": avg_loss, "metric": metric_val, **metrics}

    # ---------- Utilities / I/O ----------
    def _unpack_batch(self, batch, subgraph_sampling=False):
        """Default: batch is (inputs, labels). Override if you use other formats (e.g. torch_geometric Batch).
        Returns a tuple (inputs, labels) where `inputs` either a single object or tuple passed into model.
        """
        if not subgraph_sampling:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                return batch[0], batch[1]
            # torch_geometric.data.Batch
            try:
                # common pattern: batch.x, batch.edge_index, batch.batch
                if hasattr(batch, 'x') and hasattr(batch, 'edge_index') and hasattr(batch, 'batch'):
                    batch.x = batch.x.float()
                    batch.y = batch.y.float()
                    if getattr(batch, 'edge_attr', None) is not None:
                        batch.edge_attr = batch.edge_attr.float()
                        return (batch.x, batch.edge_index, batch.batch, batch.edge_attr), batch.y
                    else:
                        return (batch.x, batch.edge_index, batch.batch), batch.y
            except Exception as e:
                raise ValueError(f"{e}. Unknown batch format. Override _unpack_batch to handle your dataloader output.{batch}")
        
        if subgraph_sampling:
            if not hasattr(batch, 'x') and hasattr(batch,'edge_index') and hasattr(batch,'ptr') and hasattr(batch, 'y'):
                raise ValueError("Unknown batch format")
            if not hasattr(self.cfg.model_config, 'subgraph_param'):
                raise ValueError("Subgraph parameters required in config.")
            k = self.cfg.model_config.subgraph_param.k
            m = self.cfg.model_config.subgraph_param.m
            # torch_geometric.data.Batch
            try:
                if hasattr(batch, 'x') and hasattr(batch, 'edge_index') and hasattr(batch, 'batch') and hasattr(batch, 'ptr'):
                    x = batch.x.float()
                    nodes_t, edge_index_t, edge_ptr_t, graph_id_t = \
                        ugs_sampler.sample_batch(batch.edge_index.cpu(), batch.ptr.cpu(), m, k)
                    batch.y = batch.y.float()
                    return(batch.x.float(),nodes_t,edge_index_t,edge_ptr_t,graph_id_t, k), batch.y
            except Exception as e:
                # fallback: produce placeholders for all graphs in this batch (safe)
                G = int(batch.ptr.size(0) - 1)
                nodes_t, edge_index_t, edge_ptr_t, graph_id_t = self.make_placeholders(G, m, k, self.cfg.device)
                # log warn
                print(f"[warn] sampler failed; using placeholders for batch: {e}")
                return(batch.x.float(),nodes_t,edge_index_t,edge_ptr_t,graph_id_t,k), batch.y.float()


    def _get_batch_size(self, batch):
        # infer a batch-size for loss averaging. Override if needed.
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            x = batch[0]
            if hasattr(x, 'size'):
                return x.size(0)
            # for GNN Batch, batch.batch is a vector of node->graph ids
        try:
            if hasattr(batch, 'batch'):
                # number of graphs = max(batch.batch) + 1
                return int(batch.batch.max().item()) + 1
        except Exception:
            pass
        return 1

    def _to_device(self, inputs):
        # send inputs to device. `inputs` could be tuple/list or torch.Tensor or custom Batch
        if isinstance(inputs, torch.Tensor):
            return inputs.to(self.device)
        if isinstance(inputs, (list, tuple)):
            out = []
            for x in inputs:
                if isinstance(x, torch.Tensor):
                    out.append(x.to(self.device))
                else:
                    out.append(x)
            return tuple(out)
        # custom Batch object: move it on device if it has `to`
        if hasattr(inputs, 'to'):
            return inputs.to(self.device)
        return inputs
    
    def save_result(self, test, train, val):
        result_dir = 'experiment_results'
        os.makedirs(result_dir, exist_ok=True)
        save_path = os.path.join(result_dir,f"{self.cfg.name}.txt")
        with open(save_path, 'w') as f:
            f.write(f"Best model metric \n\tTest data: {test}. \n\tTrain data: {train}. \n\tVal data: {val}")

    def save_checkpoint(self, epoch: int, metric: Optional[float] = None):
        # keep only last k checkpoints
        fname = f"{self.cfg.name}_epoch{epoch:03d}.pth"
        path = os.path.join(self.cfg.checkpoint_dir, fname)
        _fname_best_model = f"best_model.pth"
        _path_best_model = os.path.join(self.cfg.checkpoint_dir, _fname_best_model)
        if (self.cfg.metric in self.down_metrics) and (self.best_metric>metric):
            _best_metric = metric
        elif (self.cfg.metric not in self.down_metrics) and (self.best_metric<metric):
            _best_metric = metric
        else:
            _best_metric = self.best_metric
        state = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optim_state': self.optimizer.state_dict(),
            'cfg': self._serialize_cfg(),
            'best_metric': _best_metric
        }
        if self.scaler is not None:
            state['scaler'] = self.scaler.state_dict()
        torch.save(state, path)

        if (self.cfg.metric in self.down_metrics) and (self.best_metric>metric):
            self.best_metric = metric
            print("Best Model Retrieved!")
            torch.save(state,_path_best_model)
        elif (self.cfg.metric not in self.down_metrics) and (self.best_metric<metric):
            self.best_metric = metric
            print("Best Model Retrieved!")
            torch.save(state,_path_best_model)
        else:
            pass

        self.logger.info("Saved checkpoint: %s", path)

        # housekeeping: remove old checkpoints keeping last K
        files = sorted([f for f in os.listdir(self.cfg.checkpoint_dir) if f.startswith(self.cfg.name)])
        if len(files) > self.cfg.keep_last_k:
            for f in files[:-self.cfg.keep_last_k]:
                try:
                    os.remove(os.path.join(self.cfg.checkpoint_dir, f))
                except Exception:
                    pass

    def _serialize_cfg(self):
        """Return a JSON/serializable-friendly representation of the ExperimentConfig.

        - replaces callable values with a short string description
        - attempts to keep nested dicts / dataclasses, but falls back to repr() where needed.
        """
        serial = {}
        for k, v in self.cfg.__dict__.items():
            # skip writer, logger, device, or other non-config runtime objects if present
            if k in ('writer', 'logger'):
                continue
            # represent callables as strings (they are not picklable)
            if callable(v):
                try:
                    name = getattr(v, "__name__", None) or repr(v)
                except Exception:
                    name = repr(v)
                serial[k] = f"<callable {name}>"
                continue

            # try common serializable types
            try:
                # simple attempt: JSON-friendly
                import json
                json.dumps(v)
                serial[k] = v
            except Exception:
                # attempt to convert dataclass / SimpleNamespace / nested dicts
                try:
                    # for objects like SimpleNamespace or dataclasses return dict of attrs
                    if hasattr(v, "__dict__"):
                        sub = {}
                        for kk, vv in v.__dict__.items():
                            if callable(vv):
                                sub[kk] = f"<callable {getattr(vv,'__name__', repr(vv))}>"
                            else:
                                try:
                                    json.dumps(vv)
                                    sub[kk] = vv
                                except Exception:
                                    sub[kk] = repr(vv)
                        serial[k] = sub
                    else:
                        serial[k] = repr(v)
                except Exception:
                    serial[k] = repr(v)
        return serial

    def load_checkpoint(self, path: str):
        self.logger.info("Loading checkpoint: %s", path)
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state['model_state'])
        self.optimizer.load_state_dict(state['optim_state'])
        if 'scaler' in state and self.scaler is not None:
            self.scaler.load_state_dict(state['scaler'])
        self.best_metric = state.get('best_metric', self.best_metric)
        return state

    def _make_placeholders(self, G, m_per_graph, k, device):
        """Return placeholder sampler outputs for G graphs (all -1s / empty edges)."""
        B_total = G * m_per_graph
        nodes_t = torch.full((B_total, k), -1, dtype=torch.long)        # CPU or device
        edge_index_t = torch.empty((2, 0), dtype=torch.long)            # no edges
        edge_ptr_t = torch.zeros((B_total + 1,), dtype=torch.long)     # all zeros -> empty blocks
        graph_id_t = torch.repeat_interleave(torch.arange(G, dtype=torch.long), torch.tensor([m_per_graph]*G))
        return nodes_t, edge_index_t, edge_ptr_t, graph_id_t