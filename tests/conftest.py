import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from fedav.config import default_config
from fedav.data.federated import build_fleet


def tiny_config(**overrides):
    """Small but structurally complete config for fast tests."""
    base = default_config(
        name="test", num_clients=6, seed=11,
        data={"samples_per_client": 60, "global_test_size": 120, "augment": False},
        train={"rounds": 2, "local_epochs": 1, "batch_size": 16},
        selection={"fraction": 0.5, "min_clients": 2},
        out_dir="/tmp/fedav-tests",
    )
    return base.merged(overrides) if overrides else base


@pytest.fixture(scope="session")
def cfg():
    return tiny_config()


@pytest.fixture(scope="session")
def fleet(cfg):
    return build_fleet(cfg)
