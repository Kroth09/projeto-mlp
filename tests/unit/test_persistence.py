import json

import numpy as np
import pytest

from mlp.exceptions import PersistenceError
from mlp.network import NeuralNetwork
from mlp.normalization import ZScoreNormalizer
from mlp.persistence import load_network, save_network


@pytest.fixture
def reference_network() -> NeuralNetwork:
    network = NeuralNetwork(
        2, [{"neurons": 2, "activation": "relu"}, {"neurons": 1, "activation": "linear"}],
        seed=42,
    )
    parameters = [([0.5, -1.0], 0.25), ([1.0, 1.0], -1.0), ([2.0, -0.5], 0.1)]
    neurons = [neuron for layer in network.layers for neuron in layer.neurons]
    for neuron, (weights, bias) in zip(neurons, parameters, strict=True):
        neuron.weights[:] = weights
        neuron.bias = bias
    network.normalizer = ZScoreNormalizer.from_dict({
        "method": "zscore", "means": [10.0, 20.0], "standard_deviations": [2.0, 5.0],
    })
    return network


def test_known_prediction_round_trip(tmp_path, reference_network) -> None:
    model_path = tmp_path / "modelo.json"
    raw = np.array([14.0, 25.0])
    before = reference_network.predict(raw)

    save_network(reference_network, model_path)
    serialized = json.loads(model_path.read_text(encoding="utf-8"))
    assert serialized["format_version"] == 1
    assert serialized["network"]["input_size"] == 2
    assert serialized["network"]["layers"][0]["weights"] == [[0.5, -1.0], [1.0, 1.0]]
    assert serialized["network"]["layers"][1]["biases"] == [0.1]

    restored = load_network(model_path)
    after = restored.predict(raw)
    repeated = restored.predict(raw)

    np.testing.assert_allclose(before, [-0.4], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(after, before)
    np.testing.assert_array_equal(repeated, after)
    np.testing.assert_array_equal(restored.forward(np.array([2.0, 1.0])), after)
    np.testing.assert_array_equal(raw, [14.0, 25.0])
    assert restored.input_size == reference_network.input_size
    assert restored.output_size == reference_network.output_size
    assert restored.normalization == reference_network.normalization
    for original, loaded in zip(reference_network.layers, restored.layers, strict=True):
        assert loaded.activation_name == original.activation_name
        assert loaded.neuron_count == original.neuron_count
        for expected, actual in zip(original.neurons, loaded.neurons, strict=True):
            np.testing.assert_array_equal(actual.weights, expected.weights)
            assert actual.bias == expected.bias


def test_model_without_normalization(tmp_path, reference_network) -> None:
    reference_network.normalizer = None
    model_path = tmp_path / "modelo.json"
    save_network(reference_network, model_path)

    data = json.loads(model_path.read_text(encoding="utf-8"))
    assert data["network"]["normalization"] is None
    restored = load_network(model_path)
    assert restored.normalizer is None
    raw = np.array([2.0, 1.0])
    np.testing.assert_array_equal(restored.predict(raw), restored.forward(raw))


def test_constant_input_uses_saved_divisor(tmp_path, reference_network) -> None:
    reference_network.normalizer = ZScoreNormalizer().fit(np.array([[7.0, 2.0], [7.0, 4.0]]))
    model_path = tmp_path / "modelo.json"
    save_network(reference_network, model_path)
    restored = load_network(model_path)

    assert restored.normalizer is not None
    np.testing.assert_array_equal(restored.normalizer.standard_deviations, [1.0, 1.0])
    raw = np.array([9.0, 4.0])
    np.testing.assert_allclose(restored.predict(raw), [-0.4], rtol=0, atol=1e-12)


def test_load_accepts_optional_metadata_and_default_zero_divisor(tmp_path, reference_network) -> None:
    model_path = tmp_path / "modelo.json"
    save_network(reference_network, model_path)
    data = json.loads(model_path.read_text(encoding="utf-8"))
    data["network"]["normalization"].pop("zero_std_divisor")
    data["network"]["input_columns"] = ["x1", "x2"]
    data["metadata"] = {"description": "fixture"}
    model_path.write_text(json.dumps(data), encoding="utf-8")

    np.testing.assert_allclose(load_network(model_path).predict(np.array([14, 25])), [-0.4])


@pytest.mark.parametrize("raw", [[1], [1, 2, 3], [[1, 2]], [float("nan"), 2], [1, float("inf")]])
def test_predict_rejects_invalid_vectors(reference_network, raw) -> None:
    with pytest.raises(ValueError, match="2 entradas|finitos"):
        reference_network.predict(np.asarray(raw))


@pytest.mark.parametrize("contents", [None, b'{"format_version":', b"\xff", b"[]"])
def test_load_reports_path_and_cause_for_invalid_file(tmp_path, contents) -> None:
    model_path = tmp_path / "modelo.json"
    if contents is not None:
        model_path.write_bytes(contents)
    with pytest.raises(PersistenceError) as failure:
        load_network(model_path)
    assert str(model_path) in str(failure.value)
    assert failure.value.__cause__ is not None


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda data: data.update(format_version=2), "Versão"),
        (lambda data: data.update(format_version=True), "Versão"),
        (lambda data: data.update(format_version=1.0), "Versão"),
        (lambda data: data.pop("network"), "network"),
        (lambda data: data["network"].update(input_size=0), "input_size"),
        (lambda data: data["network"].update(layers=[]), "layers"),
        (lambda data: data["network"].pop("normalization"), "normalization"),
        (lambda data: data["network"]["layers"][0].update(neurons=False), "neurons"),
        (lambda data: data["network"]["layers"][0].update(activation="softmax"), "ativação"),
        (lambda data: data["network"]["layers"][0].update(weights=[[0, 1], [0]]), "weights"),
        (lambda data: data["network"]["layers"][1].update(weights=[[0]]), r"Camada 2\.weights"),
        (lambda data: data["network"]["layers"][0].update(biases=[0]), "biases"),
        (lambda data: data["network"]["layers"][0].update(biases=[True, 0]), "número finito"),
        (lambda data: data["network"]["layers"][1].update(weights=[["1", 0]]), "número finito"),
        (lambda data: data["network"]["layers"][1].update(weights=[[float("inf"), 0]]), "número finito"),
        (lambda data: data["network"]["normalization"].update(means=[0]), "normalization.means"),
        (lambda data: data["network"]["normalization"].update(means=["0", 0]), "normalization.means"),
        (lambda data: data["network"]["normalization"].update(method="minmax"), "Método"),
        (lambda data: data["network"]["normalization"].update(standard_deviations=[0, 1]), "positivos"),
        (lambda data: data["network"]["normalization"].update(zero_std_divisor=True), "zero_std_divisor"),
        (lambda data: data["network"]["normalization"].update(means=[float("nan"), 0]), "NaN"),
    ],
)
def test_load_rejects_incompatible_parameters(tmp_path, reference_network, mutation, match) -> None:
    model_path = tmp_path / "modelo.json"
    save_network(reference_network, model_path)
    data = json.loads(model_path.read_text(encoding="utf-8"))
    mutation(data)
    model_path.write_text(json.dumps(data).replace("Infinity", "1e999"), encoding="utf-8")
    with pytest.raises(PersistenceError, match=match):
        load_network(model_path)


def test_invalid_save_preserves_previous_file(tmp_path, reference_network) -> None:
    model_path = tmp_path / "modelo.json"
    save_network(reference_network, model_path)
    original = model_path.read_bytes()
    reference_network.layers[0].neurons[0].weights[0] = np.inf

    with pytest.raises(PersistenceError, match="Falha ao salvar.*número finito"):
        save_network(reference_network, model_path)
    assert model_path.read_bytes() == original


def test_save_wraps_io_error(tmp_path, reference_network) -> None:
    model_path = tmp_path / "missing" / "modelo.json"
    with pytest.raises(PersistenceError) as failure:
        save_network(reference_network, model_path)
    assert str(model_path) in str(failure.value)
    assert isinstance(failure.value.__cause__, OSError)
