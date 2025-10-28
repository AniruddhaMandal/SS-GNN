# SS-GNN/src/gps/gps/cli.py
import os
import argparse
import numpy as np
from pathlib import Path
from typing import List

from gps.experiment import Experiment
from gps.config import load_config, set_config

def _resolve_config_path(user_path: str) -> Path:
    """
    Resolve --config in three ways:
    1) If absolute or exists as given, use it.
    2) If SS_GNN_CONFIG_DIR is set, resolve relative to it.
    3) If running from repo layout, look for ../../configs relative to this file.
    """
    p = Path(user_path)
    if p.is_file():
        return p

    # Env override
    base_env = os.environ.get("SS_GNN_CONFIG_DIR")
    if base_env:
        candidate = Path(base_env) / user_path
        if candidate.is_file():
            return candidate

    # Repo layout: gps/cli.py -> gps/ -> src/gps/gps/; repo root is three levels up from 'gps'
    repo_configs = Path(__file__).resolve().parents[3] / "configs"
    candidate = repo_configs / user_path
    if candidate.is_file():
        return candidate

    # As a last resort, try relative to CWD (typical when run from repo root)
    candidate = Path.cwd() / user_path
    if candidate.is_file():
        return candidate

    raise FileNotFoundError(
        f"Config file not found. Tried: '{user_path}', "
        f"$SS_GNN_CONFIG_DIR/{user_path}, '{repo_configs}/{user_path}', and CWD."
    )

def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run experiment with single or multiple seeds.")
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to config file (JSON/YAML). Accepts absolute paths, or paths relative to "
             "$SS_GNN_CONFIG_DIR, or to the repo's configs/ when running from source.",
    )
    parser.add_argument(
        "--multi-seed", "-m",
        action="store_true",
        help="Run experiment with multiple seeds (default: single run).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 10, 32, 29, 75],
        help="Custom list of seeds to use when --multi-seed is set.",
    )
    args = parser.parse_args(argv)

    cfg_path = _resolve_config_path(args.config)

    cfg = load_config(str(cfg_path))
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
