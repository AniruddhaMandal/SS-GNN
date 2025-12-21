#!/usr/bin/env python3
"""
Quick script to evaluate a checkpoint on test set.
Usage: python tools/evaluate_checkpoint.py --checkpoint path/to/best_model.pth --config path/to/config.json
"""

import argparse
import json
import torch
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'gps'))

from gps.config import set_config
from gps.experiment import Experiment


def main():
    parser = argparse.ArgumentParser(description='Evaluate a checkpoint on test set')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg_dict = json.load(f)

    # Set up experiment config
    exp_config = set_config(cfg_dict)

    # Create experiment
    exp = Experiment(exp_config)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=exp.device)
    # Try different possible keys
    if 'model_state_dict' in checkpoint:
        exp.model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model_state' in checkpoint:
        exp.model.load_state_dict(checkpoint['model_state'])
    else:
        exp.model.load_state_dict(checkpoint)
    print(f"Checkpoint loaded (epoch {checkpoint.get('epoch', 'unknown')})")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = exp.evaluate(split='test')

    # Print results
    print("\n" + "="*50)
    print("TEST SET RESULTS")
    print("="*50)
    print(f"Test Accuracy: {test_results['metric']:.5f} ({test_results['metric']*100:.2f}%)")
    print(f"Test Loss: {test_results['loss']:.5f}")
    print("="*50)

    return test_results


if __name__ == '__main__':
    main()
