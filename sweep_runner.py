#!/usr/bin/env python
"""
WandB Sweep Runner for SS-GNN experiments.

Usage:
    # Create sweep
    wandb sweep sweeps/phase2_mutag.yaml

    # Run agent
    wandb agent <sweep_id>
"""
import os
import wandb
from gps import ExperimentConfig
from gps.experiment import Experiment
from gps.config import load_config, set_config


def run_sweep():
    """Run a single sweep trial."""
    # Initialize wandb run (sweep agent expects this)
    wandb.init()

    # Get base config path from sweep config
    base_config_path = wandb.config.get('base_config', None)

    if base_config_path and os.path.exists(base_config_path):
        # Load base config
        cfg = load_config(base_config_path)
        exp_config = set_config(cfg)
    else:
        # Use default config
        exp_config = ExperimentConfig()

    # Apply sweep parameters using from_sweep
    exp_config = ExperimentConfig.from_sweep(exp_config)

    # Check for presample flag
    if wandb.config.get('presample', False):
        exp_config.presample = True

    # Update experiment name with sweep params
    k = exp_config.model_config.subgraph_param.k
    m = exp_config.model_config.subgraph_param.m
    temp = exp_config.model_config.temperature
    exp_config.name = f"sweep_k{k}_m{m}_t{temp}"

    # Run experiment (WandBWriter will detect existing wandb.run)
    experiment = Experiment(exp_config)
    result = experiment.train()

    # Log final summary metrics
    wandb.summary['final_test_acc'] = result['test_metric']
    wandb.summary['final_train_acc'] = result['train_metric']
    wandb.summary['final_val_acc'] = result['val_metric']

    print(f"Sweep trial complete: k={k}, m={m}, temp={temp}")
    print(f"  Test: {result['test_metric']:.4f}")
    print(f"  Val:  {result['val_metric']:.4f}")


if __name__ == '__main__':
    run_sweep()
