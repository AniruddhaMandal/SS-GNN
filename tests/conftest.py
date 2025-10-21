# tests/conftest.py
import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--config",
        action="store",
        help="Path to the experiment JSON config to initialize Experiment with."
    )

@pytest.fixture(scope="session")
def exp(request):
    """Boot the same Experiment as training, using the provided --config."""
    config_path = request.config.getoption("--config")
    if not config_path:
        pytest.exit("You must pass --config to pytest (e.g., --config configs/enzymes_gcn.json)")

    from gps.config import load_config, set_config
    from gps.experiment import Experiment

    cfg = load_config(config_path)
    exp_cfg = set_config(cfg)
    experiment = Experiment(exp_cfg)

    for name in ("train_loader", "val_loader", "test_loader"):
        if not hasattr(experiment, name):
            pytest.fail(f"Experiment missing {name}. Make sure Experiment exposes loaders.")

    return experiment
