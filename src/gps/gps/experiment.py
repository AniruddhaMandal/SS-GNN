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
import os, sys, warnings
from pathlib import Path
import time
import json
import logging
from typing import Any, Dict, Optional, Literal 
import random 
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import ugs_sampler
from tqdm.auto import tqdm
from . import ExperimentConfig
from . import SubgraphFeaturesBatch
from dataclasses import asdict

# ------- VSCode Debug ------
def running_in_vscode_debug():
    trace = sys.gettrace()
    return (
        (trace is not None and "debugpy" in str(trace))
        or "debugpy" in sys.modules
    )
if running_in_vscode_debug():
    warnings.simplefilter("error", UserWarning)

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
        if self.cfg.train.metric in self.down_metrics:
            self.best_metric = float('inf')
        else:
            self.best_metric = -float('inf')
        self.history = {"train_loss": [], "val_loss": []}

        # Set subgraph sampler
        self.sampler = ugs_sampler.sample_batch

        # Build components
        self.build()
        
        # Log experiment setup
        self.log_config()

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

        from gps.wandb_writer import WandBWriter, DummyWriter
        if self.cfg.tracker == 'False':
            self.writer = DummyWriter()
            return logger
        # if tracker is on initiate W&B or Tensorboard
        try:
            self.writer = WandBWriter(self.cfg)
        except ImportError:
            self.logger.info('WandB: supports wandb, `pip install wandb` and login to use.')
            try: 
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=os.path.join(self.cfg.log_dir, self.cfg.name))
                self.writer.add_hparams(self.cfg.parameter_dict(),{})
            except ImportError:
                self.logger.error('Install Tensorboard or WandB.')
                os._exit(-1)
        return logger

    def log_config(self):
        # Only log selected config sections, not everything
        key_params = {
            "experiment": self.cfg.name,
            "dataset": self.cfg.dataset_name,
            "task": self.cfg.task,
            "device": self.cfg.device,
            "seed": self.cfg.seed,
            "training": {
                "epochs": self.cfg.train.epochs,
                "batch_size": f"train={self.cfg.train.train_batch_size}, val={self.cfg.train.val_batch_size}",
                "lr": self.cfg.train.lr,
                "optimizer": self.cfg.train.optimizer,
                "weight_decay": self.cfg.train.weight_decay,
            }
        }

        # Add scheduler info if available
        if hasattr(self.cfg.train, 'scheduler') and self.cfg.train.scheduler:
            key_params["training"]["scheduler"] = self.cfg.train.scheduler.type

        pretty = json.dumps(key_params, indent=2)
        self.logger.info("=== Experiment Configuration ===")
        self.logger.info(pretty)

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
        self.logger.info("=" * 60)
        self.logger.info("Building experiment components...")
        self.logger.info("=" * 60)

        # model
        self.logger.info("→ Building model...")
        if callable(self.cfg.model_fn):
            self.model = self.cfg.model_fn(self.cfg)
        else:
            raise NotImplementedError("Provide cfg.model_fn(cfg) returning an nn.Module")
        self.model.to(self.device)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"  ✓ Model built: {total_params:,} params ({trainable_params:,} trainable)")

        # criterion
        self.logger.info("→ Building loss criterion...")
        if callable(self.cfg.criterion_fn):
            self.criterion = self.cfg.criterion_fn()
            self.logger.info(f"  ✓ Criterion: {self.criterion.__class__.__name__}")

        # dataloaders
        self.logger.info("→ Building dataloaders (this may take a while)...")
        if callable(self.cfg.dataloader_fn):
            loaders = self.cfg.dataloader_fn(self.cfg)
            if isinstance(loaders, tuple) and (2 <= len(loaders) <= 3):
                self.train_loader, self.val_loader = loaders[0], loaders[1]
                self.test_loader = loaders[2] if len(loaders) == 3 else None
                self.logger.info(f"  ✓ Train: {len(self.train_loader)} batches | Val: {len(self.val_loader)} batches" +
                               (f" | Test: {len(self.test_loader)} batches" if self.test_loader else ""))
            else:
                raise ValueError("dataloader_fn must return (train_loader, val_loader, [test_loader])")
        else:
            raise NotImplementedError("Provide cfg.dataloader_fn(cfg) returning dataloaders")

        # optimizer
        self.logger.info("→ Building optimizer...")
        self.optimizer = self._build_optimizer()
        self.logger.info(f"  ✓ Optimizer: {self.optimizer.__class__.__name__} (lr={self.cfg.train.lr})")

        # scheduler (optional)
        self.logger.info("→ Building scheduler...")
        self.scheduler = self._build_scheduler()
        if self.scheduler:
            self.logger.info(f"  ✓ Scheduler: {self.scheduler.__class__.__name__}")
        else:
            self.logger.info("  ✓ No scheduler")

        # AMP scaler
        if self.cfg.train.use_amp:
            self.logger.info("→ Enabling mixed precision training...")
            self.scaler = torch.cuda.amp.GradScaler(enabled=True)
            self.logger.info("  ✓ AMP enabled")

        # optionally resume
        if self.cfg.resume_from:
            self.logger.info(f"→ Resuming from checkpoint: {self.cfg.resume_from}")
            self.load_checkpoint(self.cfg.resume_from)

        self.logger.info("=" * 60)
        self.logger.info("✓ Build complete - Ready to train!")
        self.logger.info("=" * 60)

    def _build_optimizer(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        if self.cfg.train.optimizer.lower() in ('adam', 'adamw'):
            opt_cls = optim.AdamW if self.cfg.train.optimizer.lower() == 'adamw' else optim.Adam
            return opt_cls(params, lr=self.cfg.train.lr, weight_decay=self.cfg.train.weight_decay)
        elif self.cfg.train.optimizer.lower() == 'sgd':
            return optim.SGD(params, lr=self.cfg.train.lr, momentum=0.9, weight_decay=self.cfg.train.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {self.cfg.train.optimizer}")

    def _build_scheduler(self):
        sch = None
        s = self.cfg.train.scheduler
        if not s:
            return None
        if s.type == 'step':
            sch = optim.lr_scheduler.StepLR(self.optimizer, step_size=s.step_size, gamma=s.gamma)
        elif s.type == 'cosine':
            sch = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.cfg.train.epochs)
        elif s.type == 'reduce_on_plateau':
            sch = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=s.patience)
        else:
            self.logger.warning("Unknown scheduler type: %s", s.type)
        return sch

    # ---------- Training / evaluation ----------
    def train(self):
        """ Returns: `dict` with keys ['train_metric', 'test_metric', 'val_metric']
        """
        self.logger.info("Starting training for %d epochs", self.cfg.train.epochs)
        for epoch in range(1, self.cfg.train.epochs + 1):
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

        test_metric = test_best_stats.get('metric', None)
        train_metric = train_best_stats.get('metric', None)
        val_metric = self.best_metric

        result_out = {
            "train_metric" : train_metric,
            "test_metric" : test_metric,
            "val_metric" : val_metric
        }
        return result_out

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """Default training loop. Override for custom tasks.

        Returns a dict of statistics (must contain 'loss').
        """
        self.model.train()
        running_loss = 0.0
        n_examples = 0

        pbar = tqdm(self.train_loader, desc=f"Train {epoch}",ncols=100,dynamic_ncols=False)
        for batch in pbar:
            # Expectation: user dataloader yields a tuple (inputs..., labels)
            # You can adapt these lines to your use-case. For GNNs using torch_geometric,
            # you might receive a Batch object where `batch.x, batch.edge_index, batch.batch`
            # If subgraph sampling is required then return: 
            # `x_global`, `nodes_t`, `edge_index_t`, `edge_ptr_t`, `graph_id_t`, `k`

            batch, labels = self._unpack_batch(batch)

            batch = batch.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=self.cfg.train.use_amp):
                output = self.model(batch) 
                #user-provided criterion must accept (outputs, labels)
                if self.cfg.task == "Binary-Classification":
                    loss = self.criterion(output, labels.long())
                elif self.cfg.task == "Multi-Lable-Binary-Classification":
                    loss = self.criterion(output, labels.float())
                elif self.cfg.task == "Multi-Target-Regression":
                    loss = self.criterion(output, labels.float())
                elif self.cfg.task == "Multi-Class-Classification":
                    loss = self.criterion(output, labels.long())
                elif self.cfg.task == "Link-Prediction":
                    loss = self.criterion(output, labels)
                elif self.cfg.task == "Regression":
                    loss = self.criterion(output, labels.unsqueeze(-1))
                elif self.cfg.task == "Single-Target-Regression":
                    loss = self.criterion(output, labels.float())

                else:
                    raise ValueError(f"unknown task: {self.cfg.task}")


            # backward
            if self.cfg.train.use_amp:
                assert self.scaler is not None
                self.scaler.scale(loss).backward()
                if self.cfg.train.grad_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.cfg.train.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
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
        all_edge_label_index = []
        loaders = {
            "test": self.test_loader,
            "train": self.train_loader,
            "val": self.val_loader,
        }

        if split not in loaders:
            raise ValueError(f"Unknown split type ({split}) in evaluate.")

        loader = loaders[split]

        pbar = tqdm(loader, desc=f"{split} {epoch}" if epoch else f"{split}",ncols=86,dynamic_ncols=False)
        with torch.no_grad():
            for batch in pbar:
                batch, labels = self._unpack_batch(batch)
                batch = batch.to(self.device)
                labels = labels.to(self.device)

                with torch.amp.autocast('cuda', enabled=self.cfg.train.use_amp):
                    output = self.model(batch) 
                    #user-provided criterion must accept (outputs, labels)
                    if self.cfg.task == "Binary-Classification":
                        loss = self.criterion(output, labels.long())
                    if self.cfg.task == "Multi-Lable-Binary-Classification":
                        loss = self.criterion(output, labels.float())
                    if self.cfg.task == "Multi-Target-Regression":
                        loss = self.criterion(output, labels.float())
                    if self.cfg.task == "Multi-Class-Classification":
                        loss = self.criterion(output, labels.long())
                    if self.cfg.task == "Link-Prediction":
                        loss = self.criterion(output, labels)
                    if self.cfg.task == "Regression":
                        loss = self.criterion(output, labels.unsqueeze(-1))
                    if self.cfg.task == "Single-Target-Regression":
                        loss = self.criterion(output, labels.float())


                # collect logits and targets
                if self.cfg.task == "Link-Prediction":
                    all_logits.append(output.detach().cpu().numpy())
                    all_targets.append(labels.detach().cpu().numpy())
                    all_edge_label_index.append(batch.edge_label_index.detach().cpu().numpy())

                else:
                    all_logits.append(output.detach().cpu())
                    all_targets.append(labels.detach().cpu())

                batch_size = self._get_batch_size(batch)
                running_loss += loss.item() * batch_size
                n_examples += batch_size

        avg_loss = running_loss / max(1, n_examples)
        if self.cfg.task != 'Link-Prediction':
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
                    metrics = self.cfg.metric_fn(all_targets.numpy(), all_probs.numpy())
                if self.cfg.task == "Binary-Classification":
                    all_preds = torch.argmax(all_logits, dim=1)   # [B]
                    metrics = self.cfg.metric_fn(all_preds.numpy(), all_targets.numpy())
                if self.cfg.task == "Multi-Target-Regression":
                    metrics = self.cfg.metric_fn(all_logits.numpy(), all_targets.numpy())
                if self.cfg.task == "Multi-Class-Classification":
                    all_preds  = all_logits.argmax(dim=-1)
                    metrics = self.cfg.metric_fn(all_preds.numpy(), all_targets.numpy())
                if self.cfg.task == "Regression":
                    metrics = self.cfg.metric_fn(all_logits.numpy(), all_targets.squeeze().numpy())
                if self.cfg.task == "Link-Prediction":
                    metrics = self.cfg.metric_fn(all_logits, 
                                                 all_targets, 
                                                 all_edge_label_index)
                if self.cfg.task == "Single-Target-Regression":
                    metrics = self.cfg.metric_fn(all_logits.numpy(), all_targets.numpy())
                
            except Exception as e:
                self.logger.warning("metric_fn failed: %s", e)
                exit()
        metric_val = metrics.get(self.cfg.train.metric)

        # tracker logging
        if epoch is not None:
            self.writer.add_scalar('val/loss', avg_loss, epoch)
            if metric_val is not None:
                self.writer.add_scalar('val/metric', metric_val, epoch)

        return {"loss": avg_loss, "metric": metric_val, **metrics}

    # ---------- Utilities / I/O ----------
    def _unpack_batch(self, batch: SubgraphFeaturesBatch) -> tuple[SubgraphFeaturesBatch, torch.Tensor]:
        """
        Unpack batch data for GNN processing.
    
        Args:
            batch: Input batch (tuple, list, or torch_geometric.data.Batch)
            subgraph_sampling: Whether to perform subgraph sampling
        
        Returns:
            Tuple of (features, labels)
        """
        is_link_prediction = (self.cfg.task == 'Link-Prediction')
        edge_attr_required = (self.cfg.model_config.mpnn_type == 'gine')
        sampling_required  = self.cfg.model_config.subgraph_sampling
    
        # Validate batch format
        required_attrs = ['x', 'edge_index', 'batch']
        if edge_attr_required:
            required_attrs.append('edge_attr')
        if is_link_prediction:
            required_attrs.extend(['edge_label_index', 'edge_label'])
        else:
            required_attrs.append('y')
        if sampling_required:
            required_attrs.append('ptr')
    
        missing_attrs = [attr for attr in required_attrs if not hasattr(batch, attr)]
        if missing_attrs:
            raise ValueError(
                f"Unknown batch format. Expected attributes: {required_attrs}. "
                f"Missing: {missing_attrs}. "
                f"Override _unpack_batch to handle your dataloader output. "
                f"Received: {batch}"
            )
        # Minimal batch 
        sf_batch = SubgraphFeaturesBatch(x=batch.x.float(), 
                                         edge_index=batch.edge_index, 
                                         batch=batch.batch)
    
        # Handle optional edge attributes
        has_edge_attr = getattr(batch, 'edge_attr', None) is not None
        if has_edge_attr:
            sf_batch.edge_attr = batch.edge_attr.float()
        
        # For link prediction: labels are edge_label
        if self.cfg.task == 'Link-Prediction':
            sf_batch.edge_label_index = batch.edge_label_index
            labels = batch.edge_label.float()
        else:
            labels = batch.y.float()

        # load sample features
        if self.cfg.model_config.subgraph_sampling:
            sf_batch.ptr = batch.ptr
            sf_batch = self._sample_and_load_subgraphs(sf_batch)
        
        return (sf_batch, labels)


    def _sample_and_load_subgraphs(self, batch: SubgraphFeaturesBatch)->SubgraphFeaturesBatch:
        """Handle batch unpacking with subgraph sampling."""
        # Validate subgraph parameters
        if not getattr(self.cfg.model_config, 'subgraph_param', None):
            raise ValueError(
                "Subgraph parameters required in config. "
                "Please set cfg.model_config.subgraph_param with 'k' and 'm' values."
            )
    
        k = self.cfg.model_config.subgraph_param.k
        m = self.cfg.model_config.subgraph_param.m
    
        # Move to CPU for sampling (if needed by sampler)
        batch.edge_index = batch.edge_index.cpu()
        batch.ptr = batch.ptr.cpu()
    
        # Perform subgraph sampling
        try:
            batch.nodes_sampled, batch.edge_index_sampled, batch.edge_ptr, batch.sample_ptr, batch.edge_src_global = \
                self.sampler(batch.edge_index, batch.ptr, m, k, mode="sample")
        
        except Exception as e:
            # Fallback: use placeholders if sampling fails
            print(f"[WARNING] Subgraph sampler failed, using placeholders: {e}")
        
            num_graphs = int(batch.batch.max().item()) + 1
            batch.nodes_sampled, batch.edge_index_sampled, batch.edge_ptr, batch.sample_ptr, batch.edge_src_global = \
                self._make_placeholders(num_graphs, m, k)
        
        return batch

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



    @staticmethod
    def _safe_torch_load(path: Path, map_location):
        """Safely load a PyTorch checkpoint handling different PyTorch versions.
    
        Args:
            path: Path to checkpoint file
            map_location: Device to map tensors to
        
        Returns:
            Loaded state dictionary
        """
        # Try with weights_only=False first (for optimizer states, configs, etc.)
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            # PyTorch < 2.6 doesn't have weights_only parameter
            return torch.load(path, map_location=map_location)
        except Exception as e:
            # If we get the numpy._core error, add safe globals and retry
            if "numpy._core.multiarray.scalar" in str(e):
                try:
                    # Add numpy types to safe globals
                    import numpy as np
                    torch.serialization.add_safe_globals([
                        np.core.multiarray.scalar,
                        np.ndarray,
                        np.dtype,
                    ])
                    return torch.load(path, map_location=map_location, weights_only=False)
                except:
                    pass
            raise


    def save_checkpoint(self, epoch: int, metric: Optional[float] = None):
        """Save model checkpoint and optionally update best model.
    
        Args:
            epoch: Current training epoch
            metric: Validation metric value for this epoch
        """
        # Create checkpoint directory if it doesn't exist
        checkpoint_dir = Path(self.cfg.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
        # Determine if this is the best model
        is_best = self._is_best_metric(metric)
    
        # Update best metric if needed
        if is_best:
            self.best_metric = metric
            self.logger.info(f"New best model! Metric: {metric:.4f}")
    
        # Prepare checkpoint state
        state = self._create_checkpoint_state(epoch, metric)
    
        # Save regular checkpoint
        checkpoint_path = checkpoint_dir / f"{self.cfg.name}_epoch{epoch:03d}.pth"
        self._save_state(state, checkpoint_path)
    
        # Save best model if applicable
        if is_best:
            best_path = checkpoint_dir / "best_model.pth"
            self._save_state(state, best_path)
    
        # Clean up old checkpoints
        self._cleanup_old_checkpoints(checkpoint_dir)


    def _is_best_metric(self, metric: Optional[float]) -> bool:
        """Check if the current metric is the best so far.
    
        Args:
            metric: Current metric value
        
        Returns:
            True if this is the best metric, False otherwise
        """
        if metric is None:
            return False
    
        is_down_metric = self.cfg.train.metric in self.down_metrics
    
        if is_down_metric:
            return metric < self.best_metric
        else:
            return metric > self.best_metric


    def _create_checkpoint_state(self, epoch: int, metric: Optional[float]) -> Dict[str, Any]:
        """Create checkpoint state dictionary.
    
        Args:
            epoch: Current epoch
            metric: Current metric value
        
        Returns:
            Dictionary containing all checkpoint information
        """
        state = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optim_state': self.optimizer.state_dict(),
            'cfg': self._serialize_cfg(),
            'best_metric': self.best_metric if self._is_best_metric(metric) else self.best_metric,
            'current_metric': metric,
        }
    
        # Add gradient scaler state if using mixed precision
        if self.scaler is not None:
            state['scaler'] = self.scaler.state_dict()
    
        # Optionally add scheduler state if it exists
        if hasattr(self, 'scheduler') and self.scheduler is not None:
            state['scheduler_state'] = self.scheduler.state_dict()
    
        return state


    def _save_state(self, state: Dict[str, Any], path: Path):
        """Safely save checkpoint state to disk.
    
        Args:
            state: Checkpoint state dictionary
            path: Path to save checkpoint
        """
        try:
            # Save to temporary file first to avoid corruption
            temp_path = path.with_suffix('.pth.tmp')
            torch.save(state, temp_path)
        
            # Atomic rename
            temp_path.replace(path)
            self.logger.info(f"Checkpoint saved: {path}")
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint {path}: {e}")
            if temp_path.exists():
                temp_path.unlink()
            raise


    def _cleanup_old_checkpoints(self, checkpoint_dir: Path):
        """Remove old checkpoints, keeping only the last K.
    
        Args:
            checkpoint_dir: Directory containing checkpoints
        """
        try:
            # Find all epoch checkpoints (exclude best_model.pth)
            pattern = f"{self.cfg.name}_epoch*.pth"
            files = sorted(
                checkpoint_dir.glob(pattern),
                key=lambda p: p.stat().st_mtime
            )
        
            # Remove old checkpoints
            if len(files) > self.cfg.keep_last_k:
                for old_file in files[:-self.cfg.keep_last_k]:
                    try:
                        old_file.unlink()
                        self.logger.info(f"Removed old checkpoint: {old_file.name}")
                    except Exception as e:
                        self.logger.warning(f"Could not remove {old_file}: {e}")
        except Exception as e:
            self.logger.warning(f"Checkpoint cleanup failed: {e}")


    def _serialize_cfg(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation of the ExperimentConfig.
    
        Returns:
            Serialized configuration dictionary
        """
        def _serialize_value(v: Any) -> Any:
            """Recursively serialize a value."""
            # Handle callables
            if callable(v):
                return f"<callable {getattr(v, '__name__', repr(v))}>"
        
            # Try JSON serialization
            try:
                json.dumps(v)
                return v
            except (TypeError, ValueError):
                pass
        
            # Handle objects with __dict__
            if hasattr(v, "__dict__"):
                return {k: _serialize_value(vv) for k, vv in v.__dict__.items()}
        
            # Handle common iterables
            if isinstance(v, (list, tuple)):
                try:
                    return [_serialize_value(item) for item in v]
                except Exception:
                    return repr(v)
        
            if isinstance(v, dict):
                try:
                    return {k: _serialize_value(vv) for k, vv in v.items()}
                except Exception:
                    return repr(v)
        
            # Fallback to repr
            return repr(v)
    
        serial = {}
        excluded_keys = {'writer', 'logger', 'device'}
    
        for k, v in self.cfg.__dict__.items():
            if k in excluded_keys:
                continue
            serial[k] = _serialize_value(v)
    
        return serial


    def load_checkpoint(self, path: str, strict: bool = True, load_optimizer: bool = True):
        """Load checkpoint from disk.
    
        Args:
            path: Path to checkpoint file
            strict: Whether to strictly enforce state dict loading
            load_optimizer: Whether to load optimizer state
        
        Returns:
            Loaded checkpoint state dictionary
        """
        path = Path(path)
    
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
    
        self.logger.info(f"Loading checkpoint: {path}")
    
        try:
            state = self._safe_torch_load(path, self.device)
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            raise
    
        # Load model state
        try:
            self.model.load_state_dict(state['model_state'], strict=strict)
        except Exception as e:
            self.logger.error(f"Failed to load model state: {e}")
            if strict:
                raise
            self.logger.warning("Continuing with partial model loading...")
    
        # Load optimizer state
        if load_optimizer and 'optim_state' in state:
            try:
                self.optimizer.load_state_dict(state['optim_state'])
            except Exception as e:
                self.logger.warning(f"Failed to load optimizer state: {e}")
    
        # Load gradient scaler state
        if 'scaler' in state and self.scaler is not None:
            try:
                self.scaler.load_state_dict(state['scaler'])
            except Exception as e:
                self.logger.warning(f"Failed to load scaler state: {e}")
    
        # Load scheduler state if available
        if 'scheduler_state' in state and hasattr(self, 'scheduler') and self.scheduler is not None:
            try:
                self.scheduler.load_state_dict(state['scheduler_state'])
            except Exception as e:
                self.logger.warning(f"Failed to load scheduler state: {e}")
    
        # Update best metric
        self.best_metric = state.get('best_metric', self.best_metric)
    
        epoch = state.get('epoch', 0)
        self.logger.info(f"Checkpoint loaded successfully (epoch {epoch})")
    
        return state

    def _make_placeholders(self, G, m_per_graph, k):
        """Return placeholder sampler outputs for G graphs (all -1s / empty edges)."""
        B_total = G * m_per_graph
        nodes_t = torch.full((B_total, k), -1, dtype=torch.long)        # CPU or device
        edge_index_t = torch.empty((2, 0), dtype=torch.long)            # no edges
        edge_ptr_t = torch.zeros((B_total + 1,), dtype=torch.long)     # all zeros -> empty blocks
        sample_ptr_t = torch.zeros((G+1), dtype=torch.long)
        edge_src_global_t = torch.empty((0), dtype=torch.long)
        return nodes_t, edge_index_t, edge_ptr_t, sample_ptr_t, edge_src_global_t