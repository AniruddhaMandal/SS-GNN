# sweep.py

import wandb

import json
from functools import partial
from gps import ExperimentConfig
from gps.experiment import Experiment
from gps.config import load_config, set_config

def spawn_sweep(project_name):
    """Initialize and run sweep"""
    # Load sweep configuration from YAML file
    config_path = 'sweep_config.json'
    with open(config_path, 'r') as f:
        sweep_config = json.load(f)
    # Create sweep
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project=project_name
    )
    return sweep_id
    
def train_for_sweep(cfg: ExperimentConfig):
    """Wrapper function for sweep agent"""

    # Initialize wandb run (sweep agent will set config)
    run = wandb.init()
    
    # Setup experiment config (call after wandb.inti)
    cfg = ExperimentConfig.from_sweep(cfg)

    # Create experiment with sweep config
    experiment = Experiment(cfg)
    
    # Run training
    test_results = experiment.train()
    
    # Log final results
    wandb.log(test_results)
    
    run.finish()
    return test_results

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run hyperparameter sweep')
    parser.add_argument('--project', type=str, default='SS-GNN-Tune',
                        help='W&B project name')
    parser.add_argument('--entity', type=str, help='W&B entity name')
    parser.add_argument('--count', type=int, default=50,
                        help='Number of sweep runs')
    parser.add_argument('--sweep-id', type=str, default=None,
                        help='Existing sweep ID to continue')
    parser.add_argument('--config', '-c', type=str, default='../configs/ss_gnn/TUData/gin-proteins.json',
                        help='Experiment config json file for tuning.')
    
    args = parser.parse_args()

    cfg = load_config(args.config)
    exp_cfg = set_config(cfg)
    train_fn = partial(train_for_sweep,cfg=exp_cfg)
    
    if args.sweep_id:
        sweep_id = args.sweep_id
        print(f"Continuing sweep: {sweep_id}")
    else:
        sweep_id = spawn_sweep(project_name=args.project) # spawn new sweep

    wandb.agent(
        sweep_id, 
        function=train_fn, 
        count=args.count, 
        entity=args.entity, 
        project=args.project)
