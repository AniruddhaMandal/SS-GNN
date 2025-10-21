import argparse
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
        stats = []
        for seed in args.seeds:
            exp_config.seed = seed
            experiment = Experiment(exp_config)
            result = experiment.train()
            stats.append(result)
            print(f"Seed {seed}: {result:.5f}")
        stats = np.array(stats)
        print(f"\nFinal result over {len(args.seeds)} seeds: {stats.mean():.5f} ± {stats.std():.5f}")

    else:
        print(f"Running single experiment with seed {exp_config.seed}")
        experiment = Experiment(exp_config)
        result = experiment.train()
        print(f"Result: {result:.5f}")
