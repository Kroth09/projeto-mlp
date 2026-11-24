"""Gera evidências reproduzíveis do núcleo matemático da MLP."""

from __future__ import annotations

import numpy as np

from mlp.config import LayerConfig
from mlp.metrics import mean_squared_error, mean_squared_error_for_dataset
from mlp.network import NeuralNetwork


XOR_INPUTS = np.array(
    [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
    dtype=np.float64,
)
XOR_OUTPUTS = np.array([[0.0], [1.0], [1.0], [0.0]], dtype=np.float64)


def manual_forward_result() -> float:
    """Executa o exemplo manual 2 -> 2 -> 1 do guia."""
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
    return float(network.forward(np.array([2.0, 1.0]))[0])


def manual_update_result() -> tuple[float, float]:
    """Executa a atualização manual 1 -> 1 do guia."""
    network = NeuralNetwork(1, [LayerConfig(1, "linear")], seed=42)
    neuron = network.layers[0].neurons[0]
    neuron.weights[:] = [0.5]
    neuron.bias = 0.1
    network.forward(np.array([2.0]))
    network.backward(np.array([2.0]), learning_rate=0.1)
    return float(neuron.weights[0]), neuron.bias


def maximum_gradient_relative_error() -> float:
    """Compara todos os parâmetros de uma rede pequena por diferenças centrais."""
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
                analytical_value = analytical[layer_index].weight_gradients[
                    neuron_index, weight_index
                ]
                relative_errors.append(
                    abs(analytical_value - numerical)
                    / max(1.0, abs(analytical_value), abs(numerical))
                )

            original_bias = neuron.bias
            neuron.bias = original_bias + epsilon
            error_plus = current_error()
            neuron.bias = original_bias - epsilon
            error_minus = current_error()
            neuron.bias = original_bias
            numerical = (error_plus - error_minus) / (2.0 * epsilon)
            analytical_value = analytical[layer_index].bias_gradients[neuron_index]
            relative_errors.append(
                abs(analytical_value - numerical)
                / max(1.0, abs(analytical_value), abs(numerical))
            )
    return max(relative_errors)


def xor_result() -> tuple[float, float, np.ndarray]:
    """Treina a rede XOR com SGD e retorna erros e previsões."""
    seed = 42
    network = NeuralNetwork(
        2,
        [LayerConfig(4, "tanh"), LayerConfig(1, "sigmoid")],
        seed=seed,
    )
    initial_mse = mean_squared_error_for_dataset(network, XOR_INPUTS, XOR_OUTPUTS)
    order_rng = np.random.default_rng(seed)
    for _ in range(10_000):
        for index in order_rng.permutation(len(XOR_INPUTS)):
            network.forward(XOR_INPUTS[index])
            network.backward(XOR_OUTPUTS[index], learning_rate=0.1)
    final_mse = mean_squared_error_for_dataset(network, XOR_INPUTS, XOR_OUTPUTS)
    predictions = np.asarray(
        [network.predict(row)[0] for row in XOR_INPUTS], dtype=np.float64
    )
    return initial_mse, final_mse, predictions


def main() -> None:
    forward_result = manual_forward_result()
    weight, bias = manual_update_result()
    gradient_error = maximum_gradient_relative_error()
    initial_mse, final_mse, predictions = xor_result()

    print(f"Forward manual 2 -> 2 -> 1: {forward_result:.12f}")
    print(f"Atualização manual 1 -> 1: peso={weight:.12f}, viés={bias:.12f}")
    print(f"Maior erro relativo do gradient check: {gradient_error:.12e}")
    print(f"XOR MSE inicial: {initial_mse:.12f}")
    print(f"XOR MSE final: {final_mse:.12f}")
    print(f"XOR previsões: {predictions.tolist()}")


if __name__ == "__main__":
    main()

