import numpy as np
import pytest

from mlp.config import LayerConfig
from mlp.metrics import mean_squared_error
from mlp.network import NeuralNetwork


def test_manual_single_neuron_update() -> None:
    network = NeuralNetwork(1, [LayerConfig(1, "linear")], seed=1)
    neuron = network.layers[0].neurons[0]
    neuron.weights[:] = [0.5]
    neuron.bias = 0.1
    inputs = np.array([2.0])
    expected = np.array([2.0])

    before = mean_squared_error(network.forward(inputs), expected)
    network.backward(expected, learning_rate=0.1)

    assert neuron.weights[0] == pytest.approx(0.86)
    assert neuron.bias == pytest.approx(0.28)
    after = mean_squared_error(network.forward(inputs), expected)
    assert after < before


def test_hidden_deltas_use_weights_from_forward() -> None:
    network = NeuralNetwork(
        2,
        [LayerConfig(2, "linear"), LayerConfig(1, "linear")],
        seed=1,
    )
    hidden, output = network.layers
    hidden.neurons[0].weights[:] = [0.5, -0.2]
    hidden.neurons[0].bias = 0.1
    hidden.neurons[1].weights[:] = [0.3, 0.8]
    hidden.neurons[1].bias = -0.1
    output.neurons[0].weights[:] = [0.7, -0.4]
    output.neurons[0].bias = 0.2
    original_weights = [
        neuron.weights.copy() for layer in network.layers for neuron in layer.neurons
    ]

    network.forward(np.array([2.0, 1.0]))
    gradients = network.compute_gradients(np.array([0.5]))

    assert output.neurons[0].delta == pytest.approx(-0.38)
    assert hidden.neurons[0].delta == pytest.approx(-0.266)
    assert hidden.neurons[1].delta == pytest.approx(0.152)
    current_weights = [
        neuron.weights for layer in network.layers for neuron in layer.neurons
    ]
    for original, current in zip(original_weights, current_weights, strict=True):
        np.testing.assert_array_equal(original, current)
    np.testing.assert_allclose(gradients[0].weight_gradients[0], [-0.532, -0.266])


def test_output_delta_uses_two_over_output_count() -> None:
    network = NeuralNetwork(1, [LayerConfig(2, "linear")], seed=1)
    layer = network.layers[0]
    layer.neurons[0].weights[:] = [0.0]
    layer.neurons[0].bias = 1.0
    layer.neurons[1].weights[:] = [0.0]
    layer.neurons[1].bias = 3.0

    network.forward(np.array([5.0]))
    network.compute_deltas(np.array([2.0, 1.0]))

    assert layer.neurons[0].delta == pytest.approx(-1.0)
    assert layer.neurons[1].delta == pytest.approx(2.0)


def test_backward_requires_forward() -> None:
    network = NeuralNetwork(1, [LayerConfig(1, "linear")], seed=1)
    with pytest.raises(RuntimeError, match="forward"):
        network.backward(np.array([0.0]), learning_rate=0.1)


@pytest.mark.parametrize("learning_rate", [0.0, -0.1, float("inf")])
def test_invalid_learning_rate_is_rejected(learning_rate: float) -> None:
    network = NeuralNetwork(1, [LayerConfig(1, "linear")], seed=1)
    network.forward(np.array([1.0]))
    with pytest.raises(ValueError, match="learning_rate"):
        network.backward(np.array([0.0]), learning_rate=learning_rate)


def test_second_update_requires_new_forward() -> None:
    network = NeuralNetwork(1, [LayerConfig(1, "linear")], seed=1)
    network.forward(np.array([1.0]))
    network.backward(np.array([0.0]), learning_rate=0.1)

    with pytest.raises(RuntimeError, match="forward"):
        network.backward(np.array([0.0]), learning_rate=0.1)

