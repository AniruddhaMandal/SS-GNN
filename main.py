import os
import json
import argparse
import numpy as np
from gps.experiment import Experiment
from gps.config import load_config, set_config

def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """
    Overrides fields in cfg (dict) based on key=value pairs.
    Supports nested keys like 'train.lr=0.01'.
    Modifies cfg in place and also returns it.
    """
    for ov in overrides:
        if '=' not in ov:
            raise ValueError(f"Invalid override: {ov}. Must be key=value.")
        key, value = ov.split('=', 1)

        # Convert string to number/bool/null if possible
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass  # keep as string

        # Walk down nested dict keys
        parts = key.split('.')
        obj = cfg
        for p in parts[:-1]:
            if p not in obj or not isinstance(obj[p], dict):
                print(f"not present: {p}")
                obj[p] = {}  # create intermediate levels if missing
            obj = obj[p]
        obj[parts[-1]] = value
    return cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiment with single or multiple seeds.")
    parser.add_argument('--config', '-c',
                        type=str,
                        required=True,
                        help='Path to config JSON file.')
    parser.add_argument('--multi-seed', '-m',
                        action='store_true',
                        help='Run experiment with multiple seeds (default: single run).')
    parser.add_argument('--seeds',
                        nargs='+',
                        type=int,
                        default=[42, 10, 32, 29, 75],
                        help='Custom list of seeds to use when --multi-seed is set.')
    parser.add_argument('--override', '-o', nargs='*', default=[],
                        help='Override config values, e.g. train.lr=0.01 model.hidden_dim=128')
    parser.add_argument('--presample', '-p',
                        action='store_true',
                        help='Presample subgraphs once before training for computational speed.')
    args = parser.parse_args()

    # Load and set config
    cfg = load_config(args.config)

    # Apply commandline overrides
    if args.override:
        cfg = apply_overrides(cfg, args.override)

    exp_config = set_config(cfg)

    # Set presample flag from CLI
    if args.presample:
        exp_config.presample = True

    if args.multi_seed:
        print(f"Running experiment with multiple seeds: {args.seeds}")
        train_metrics = []
        test_metrics = []
        val_metrics = []
        for seed in args.seeds:
            exp_config.seed = seed
            experiment = Experiment(exp_config)
            result = experiment.train()
            train_metrics.append(result['train_metric'])
            test_metrics.append(result['test_metric'])
            val_metrics.append(result['val_metric'])
        train_metrics = np.array(train_metrics)
        test_metrics = np.array(test_metrics)
        val_metrics = np.array(val_metrics)
        out_str = f"Final result over {len(args.seeds)} seeds:\
              \n\t Test: {test_metrics.mean():.5f} ± {test_metrics.std():.5f}\
              \n\t Train: {train_metrics.mean():.5f} ± {train_metrics.std():.5f}\
              \n\t Val: {val_metrics.mean():.5f} ± {val_metrics.std():.5f}\
              "
        print(out_str)
    else:
        print(f"Running single experiment with seed {exp_config.seed}")
        experiment = Experiment(exp_config)
        result = experiment.train()
        train_metrics = np.array(result['train_metric'])
        test_metrics = np.array(result['test_metric'])
        val_metrics = np.array(result['val_metric'])
        out_str = f"Result: \
                    \n\tTest: {test_metrics:.5f}\
                    \n\tTrain: {train_metrics:.5f}\
                    \n\tVal: {val_metrics:.5f}"
        print(out_str)

    os.makedirs('experiment_results',exist_ok=True)
    with open(f"experiment_results/{exp_config.name}.txt",'w') as f:
        f.write(out_str)
