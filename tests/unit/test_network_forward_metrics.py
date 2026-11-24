import numpy as np
import pytest

from mlp.config import LayerConfig
from mlp.metrics import mean_squared_error, mean_squared_error_for_dataset
from mlp.network import NeuralNetwork


def build_manual_network() -> NeuralNetwork:
    network = NeuralNetwork(
        2,
        [LayerConfig(2, "linear"), LayerConfig(1, "linear")],
        seed=42,
    )
    hidden, output = network.layers
    hidden.neurons[0].weights[:] = [0.5, -0.2]
    hidden.neurons[0].bias = 0.1
    hidden.neurons[1].weights[:] = [0.3, 0.8]
    hidden.neurons[1].bias = -0.1
    output.neurons[0].weights[:] = [0.7, -0.4]
    output.neurons[0].bias = 0.2
    return network


def test_manual_two_layer_forward() -> None:
    network = build_manual_network()

    prediction = network.forward(np.array([2.0, 1.0]))

    np.testing.assert_allclose(prediction, [0.31], atol=1e-12)


def test_dynamic_architecture_and_seed_reproducibility() -> None:
    config = [
        LayerConfig(5, "tanh"),
        LayerConfig(3, "relu"),
        LayerConfig(2, "sigmoid"),
    ]
    first = NeuralNetwork(4, config, seed=11)
    second = NeuralNetwork(4, config, seed=11)

    assert [layer.input_size for layer in first.layers] == [4, 5, 3]
    assert [layer.neuron_count for layer in first.layers] == [5, 3, 2]
    for first_layer, second_layer in zip(first.layers, second.layers, strict=True):
        for first_neuron, second_neuron in zip(
            first_layer.neurons, second_layer.neurons, strict=True
        ):
            np.testing.assert_array_equal(first_neuron.weights, second_neuron.weights)


def test_network_rejects_invalid_input_shape() -> None:
    network = NeuralNetwork(2, [LayerConfig(1, "linear")], seed=1)
    with pytest.raises(ValueError, match="esperava 2 entradas"):
        network.forward(np.array([1.0]))


def test_mean_squared_error_matches_manual_value() -> None:
    result = mean_squared_error(np.array([1.0, 3.0]), np.array([2.0, 1.0]))
    assert result == pytest.approx(2.5)


def test_mean_squared_error_rejects_different_shapes() -> None:
    with pytest.raises(ValueError, match="formatos diferentes"):
        mean_squared_error(np.array([1.0]), np.array([[1.0]]))


def test_dataset_mse_includes_all_examples_and_outputs() -> None:
    network = NeuralNetwork(1, [LayerConfig(1, "linear")], seed=1)
    network.layers[0].neurons[0].weights[:] = [2.0]
    network.layers[0].neurons[0].bias = 0.0
    inputs = np.array([[1.0], [2.0], [3.0]])
    expected = np.array([[2.0], [5.0], [5.0]])

    result = mean_squared_error_for_dataset(network, inputs, expected)

    assert result == pytest.approx((0.0 + 1.0 + 1.0) / 3.0)

