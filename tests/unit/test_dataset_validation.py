import numpy as np
import pytest

from mlp.config import LayerConfig
from mlp.dataset import SupervisedDataset, validate_dataset_compatibility
from mlp.exceptions import DatasetError
from mlp.network import NeuralNetwork


def make_dataset(input_size: int, output_size: int) -> SupervisedDataset:
    return SupervisedDataset(
        np.zeros((3, input_size)),
        np.zeros((3, output_size)),
        tuple(f"x{index}" for index in range(input_size)),
        tuple(f"y{index}" for index in range(output_size)),
    )


def test_accepts_compatible_dataset() -> None:
    dataset = make_dataset(2, 1)
    network = NeuralNetwork(2, [LayerConfig(3, "tanh"), LayerConfig(1, "linear")])
    validate_dataset_compatibility(dataset, network)


def test_rejects_incompatible_input_count() -> None:
    dataset = make_dataset(3, 1)
    network = NeuralNetwork(2, [LayerConfig(1, "linear")])
    with pytest.raises(DatasetError, match="3 entradas.*espera 2"):
        validate_dataset_compatibility(dataset, network)


def test_rejects_incompatible_output_count() -> None:
    dataset = make_dataset(2, 2)
    network = NeuralNetwork(2, [LayerConfig(1, "linear")])
    with pytest.raises(DatasetError, match="2 saídas.*produz 1"):
        validate_dataset_compatibility(dataset, network)


def test_dataset_copies_arrays_and_validates_pairing() -> None:
    inputs = np.array([[1.0], [2.0]])
    outputs = np.array([[10.0], [20.0]])
    dataset = SupervisedDataset(inputs, outputs, ("x",), ("y",))
    inputs[0, 0] = 999.0
    assert dataset.inputs[0, 0] == 1.0

    with pytest.raises(DatasetError, match="mesma quantidade"):
        SupervisedDataset(np.zeros((2, 1)), np.zeros((3, 1)), ("x",), ("y",))

