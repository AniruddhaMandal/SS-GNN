import argparse
import numpy as np
from gps.experiment import Experiment
from gps.config import load_config, set_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import config file.")
    parser.add_argument('--config', '-c',
                        type=str,
                        required=True,
                        help='Insert a config as a json file.')
    args = parser.parse_args()
    cfg = load_config(args.config)

    exp_config = set_config(cfg)
    seeds = [42, 10, 32, 29, 75, 53]
    stats = []
    for seed in seeds:
        exp_config.seed = seed
        experiment = Experiment(exp_config)
        stats.append(experiment.train())
    stats = np.array(stats)
    print(stats)
    print(stats.mean(), stats.std())