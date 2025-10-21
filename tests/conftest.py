# tests/conftest.py
import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--config",
        action="store",
        help="Path to the experiment JSON config to initialize Experiment with."
    )

@pytest.fixture(scope="session")
def config_path(request):
    path = request.config.getoption("--config")
    if not path:
        pytest.exit("You must pass --config / load plugin before using config_path")
    return path

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

    # If your loaders are built lazily, uncomment:
    # if not hasattr(experiment, "train_loader"):
    #     experiment.setup_data()

    for name in ("train_loader", "val_loader", "test_loader"):
        if not hasattr(experiment, name):
            pytest.fail(f"Experiment missing {name}. Make sure Experiment exposes loaders.")

    return experiment

@pytest.fixture(scope="session")
def data_loaders(exp):
    """Return the same train/val/test loaders used by the experiment."""
    return {
        "train": exp.train_loader,
        "val":   exp.val_loader,
        "test":  exp.test_loader,
    }
