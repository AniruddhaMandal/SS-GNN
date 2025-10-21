import pytest

def test_one_epoch_reproducibility(request, exp):
    config_path = request.config.getoption("--config")
    if not config_path:
        pytest.exit("You must pass --config to pytest (e.g., --config configs/enzymes_gcn.json)")

    from gps.config import load_config, set_config
    from gps.experiment import Experiment

    # second experiment for comparison
    cfg_prime = load_config(config_path)
    exp_cfg_prime = set_config(cfg_prime)
    exp_prime = Experiment(exp_cfg_prime)

    # train both one epoch
    exp.train_one_epoch(0)
    exp_prime.train_one_epoch(0)

    # compare metric on validation set
    a = exp.evaluate().get('metric')
    b = exp_prime.evaluate().get('metric')
    assert a == b, "Not reproducing one epoch"