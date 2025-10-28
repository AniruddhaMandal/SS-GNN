import argparse
import os
import numpy as np
from gps.experiment import Experiment
from gps.config import load_config, set_config

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
    args = parser.parse_args()

    # Load and set config
    cfg = load_config(args.config)
    exp_config = set_config(cfg)

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
        out_str = f"\nFinal result over {len(args.seeds)} seeds:\
              \n\t Test: {test_metrics.mean():.5f} ± {test_metrics.std():.5f}\
              \n\t Train: {train_metrics.mean():.5f} ± {train_metrics.std():.5f}\
              \n\t Val: {val_metrics.mean():.5f} ± {val_metrics.std():.5f}\
              "
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
    with open(f"experiment_results/{exp_config.name}",'w') as f:
        f.write(out_str)
