import numpy as np

from mlp.config import LayerConfig
from mlp.metrics import mean_squared_error
from mlp.network import NeuralNetwork


def test_analytical_gradients_match_central_differences() -> None:
    network = NeuralNetwork(
        2,
        [LayerConfig(2, "tanh"), LayerConfig(1, "sigmoid")],
        seed=42,
    )
    hidden, output = network.layers
    hidden.neurons[0].weights[:] = [0.1, -0.2]
    hidden.neurons[0].bias = 0.05
    hidden.neurons[1].weights[:] = [0.4, 0.3]
    hidden.neurons[1].bias = -0.15
    output.neurons[0].weights[:] = [0.7, -0.5]
    output.neurons[0].bias = 0.2
    inputs = np.array([0.6, -0.1])
    expected = np.array([0.8])

    network.forward(inputs)
    analytical = network.compute_gradients(expected)
    epsilon = 1e-5
    relative_errors: list[float] = []

    def current_error() -> float:
        return mean_squared_error(network.forward(inputs), expected)

    for layer_index, layer in enumerate(network.layers):
        for neuron_index, neuron in enumerate(layer.neurons):
            for weight_index in range(neuron.weights.size):
                original = float(neuron.weights[weight_index])
                neuron.weights[weight_index] = original + epsilon
                error_plus = current_error()
                neuron.weights[weight_index] = original - epsilon
                error_minus = current_error()
                neuron.weights[weight_index] = original
                numerical = (error_plus - error_minus) / (2.0 * epsilon)
                expected_gradient = analytical[layer_index].weight_gradients[
                    neuron_index, weight_index
                ]
                relative_errors.append(
                    abs(expected_gradient - numerical)
                    / max(1.0, abs(expected_gradient), abs(numerical))
                )

            original_bias = neuron.bias
            neuron.bias = original_bias + epsilon
            error_plus = current_error()
            neuron.bias = original_bias - epsilon
            error_minus = current_error()
            neuron.bias = original_bias
            numerical = (error_plus - error_minus) / (2.0 * epsilon)
            expected_gradient = analytical[layer_index].bias_gradients[neuron_index]
            relative_errors.append(
                abs(expected_gradient - numerical)
                / max(1.0, abs(expected_gradient), abs(numerical))
            )

    assert max(relative_errors) < 1e-5

