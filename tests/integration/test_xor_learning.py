import numpy as np

from mlp.config import LayerConfig
from mlp.metrics import mean_squared_error_for_dataset
from mlp.network import NeuralNetwork


XOR_INPUTS = np.array(
    [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
    dtype=np.float64,
)
XOR_OUTPUTS = np.array([[0.0], [1.0], [1.0], [0.0]], dtype=np.float64)


def train_xor(seed: int, epochs: int) -> tuple[NeuralNetwork, float, float]:
    network = NeuralNetwork(
        2,
        [LayerConfig(4, "tanh"), LayerConfig(1, "sigmoid")],
        seed=seed,
    )
    initial_mse = mean_squared_error_for_dataset(network, XOR_INPUTS, XOR_OUTPUTS)
    order_rng = np.random.default_rng(seed)
    for _ in range(epochs):
        for index in order_rng.permutation(len(XOR_INPUTS)):
            network.forward(XOR_INPUTS[index])
            network.backward(XOR_OUTPUTS[index], learning_rate=0.1)
    final_mse = mean_squared_error_for_dataset(network, XOR_INPUTS, XOR_OUTPUTS)
    return network, initial_mse, final_mse


def test_network_learns_xor() -> None:
    network, initial_mse, final_mse = train_xor(seed=42, epochs=10_000)
    predictions = np.array(
        [network.predict(inputs)[0] for inputs in XOR_INPUTS], dtype=np.float64
    )

    assert final_mse < initial_mse
    assert predictions[0] < 0.2
    assert predictions[1] > 0.8
    assert predictions[2] > 0.8
    assert predictions[3] < 0.2


def test_fixed_seed_and_order_reproduce_training() -> None:
    first, _, _ = train_xor(seed=7, epochs=250)
    second, _, _ = train_xor(seed=7, epochs=250)

    for first_layer, second_layer in zip(first.layers, second.layers, strict=True):
        for first_neuron, second_neuron in zip(
            first_layer.neurons, second_layer.neurons, strict=True
        ):
            np.testing.assert_array_equal(first_neuron.weights, second_neuron.weights)
            assert first_neuron.bias == second_neuron.bias

