import argparse
from gps.config import load_config, set_config, build_model,\
      build_dataloader, build_metric, build_criterion
from gps.experiment import Experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import config file.")
    parser.add_argument('--config', '-c',
                        type=str,
                        required=True,
                        help='Insert a config as a json file.')
    args = parser.parse_args()
    cfg = load_config(args.config)

    exp_config = set_config(cfg)
    print(exp_config.model_fn)
    experiment = Experiment(exp_config)
    experiment.train()