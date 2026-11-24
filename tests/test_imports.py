import importlib

import mlp


def test_package_can_be_imported() -> None:
    assert mlp.__version__ == "0.1.0"


def test_public_modules_can_be_imported() -> None:
    module_names = [
        "activations",
        "cli",
        "config",
        "data_pipeline",
        "dataset",
        "exceptions",
        "initialization",
        "layer",
        "metrics",
        "network",
        "neuron",
        "normalization",
        "persistence",
        "trainer",
        "types",
    ]
    for module_name in module_names:
        importlib.import_module(f"mlp.{module_name}")

