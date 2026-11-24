import numpy as np
import pytest

from mlp.initialization import (
    initialize_biases,
    initialize_he,
    initialize_xavier,
)
from mlp.layer import Layer


def test_xavier_is_reproducible_and_has_expected_shape() -> None:
    first = initialize_xavier(5, 3, np.random.default_rng(42))
    second = initialize_xavier(5, 3, np.random.default_rng(42))
    different = initialize_xavier(5, 3, np.random.default_rng(43))

    np.testing.assert_array_equal(first, second)
    assert first.shape == (3, 5)
    assert first.dtype == np.float64
    assert not np.array_equal(first, different)
    assert not np.all(first == first.flat[0])


def test_he_has_expected_statistical_scale() -> None:
    fan_in = 100
    weights = initialize_he(fan_in, 2000, np.random.default_rng(7))
    expected_std = np.sqrt(2.0 / fan_in)

    assert weights.std() == pytest.approx(expected_std, rel=0.03)


def test_biases_start_at_zero() -> None:
    biases = initialize_biases(4)
    np.testing.assert_array_equal(biases, np.zeros(4, dtype=np.float64))


@pytest.mark.parametrize(("fan_in", "fan_out"), [(0, 2), (2, 0), (-1, 2)])
def test_invalid_sizes_are_rejected(fan_in: int, fan_out: int) -> None:
    with pytest.raises(ValueError):
        initialize_xavier(fan_in, fan_out, np.random.default_rng(1))


def test_layer_selects_he_for_relu() -> None:
    layer = Layer(3, 2, "relu", np.random.default_rng(9))
    expected = initialize_he(3, 2, np.random.default_rng(9))

    actual = np.vstack([neuron.weights for neuron in layer.neurons])
    assert layer.initialization_method == "he"
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("activation", ["linear", "sigmoid", "tanh"])
def test_layer_selects_xavier_for_non_relu(activation: str) -> None:
    layer = Layer(3, 2, activation, np.random.default_rng(9))
    expected = initialize_xavier(3, 2, np.random.default_rng(9))

    actual = np.vstack([neuron.weights for neuron in layer.neurons])
    assert layer.initialization_method == "xavier"
    np.testing.assert_array_equal(actual, expected)


def test_public_initializer_rejects_invalid_rng() -> None:
    with pytest.raises(TypeError, match="Generator"):
        initialize_xavier(2, 2, None)


def test_layer_rejects_invalid_activation_type() -> None:
    with pytest.raises(ValueError, match="string não vazia"):
        Layer(2, 2, None, np.random.default_rng(1))

