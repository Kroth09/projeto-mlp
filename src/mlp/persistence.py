"""Persistência versionada da rede neural em JSON."""

import json
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np

from mlp.activations import SUPPORTED_ACTIVATIONS
from mlp.config import LayerConfig
from mlp.exceptions import NormalizationError, PersistenceError
from mlp.network import NeuralNetwork
from mlp.normalization import ZScoreNormalizer
from mlp.types import FloatArray


FORMAT_VERSION = 1


@dataclass(frozen=True)
class _SavedLayer:
    config: LayerConfig
    weights: FloatArray
    biases: FloatArray


@dataclass(frozen=True)
class _SavedNetwork:
    input_size: int
    layers: list[_SavedLayer]
    normalizer: ZScoreNormalizer | None
    input_columns: tuple[str, ...] | None
    output_columns: tuple[str, ...] | None


def _object(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PersistenceError(f"{field} deve ser um objeto JSON.")
    return value


def _positive_int(value: object, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise PersistenceError(f"{field} deve ser um inteiro positivo.")
    return value


def _number(value: object, field: str) -> float:
    if type(value) not in (int, float):
        raise PersistenceError(f"{field} deve ser um número finito.")
    try:
        number = float(value)
    except OverflowError as error:
        raise PersistenceError(f"{field} deve ser um número finito.") from error
    if not isfinite(number):
        raise PersistenceError(f"{field} deve ser um número finito.")
    return number


def _vector(value: object, size: int, field: str) -> list[float]:
    if not isinstance(value, list) or len(value) != size:
        raise PersistenceError(f"{field} deve conter {size} valores.")
    return [_number(item, f"{field}[{index}]") for index, item in enumerate(value)]


def _layer(value: object, fan_in: int, index: int) -> _SavedLayer:
    field = f"Camada {index}"
    data = _object(value, field)
    neurons = _positive_int(data.get("neurons"), f"{field}.neurons")
    activation = data.get("activation")
    if not isinstance(activation, str) or activation not in SUPPORTED_ACTIVATIONS:
        raise PersistenceError(f"{field} tem ativação desconhecida: {activation!r}.")

    weights = data.get("weights")
    if not isinstance(weights, list) or len(weights) != neurons or any(
        not isinstance(row, list) or len(row) != fan_in for row in weights
    ):
        raise PersistenceError(f"{field}.weights deve ter formato ({neurons}, {fan_in}).")
    weight_rows = [
        _vector(row, fan_in, f"{field}.weights[{row_index}]")
        for row_index, row in enumerate(weights)
    ]
    biases = _vector(data.get("biases"), neurons, f"{field}.biases")
    return _SavedLayer(
        LayerConfig(neurons, activation),
        np.asarray(weight_rows, dtype=np.float64),
        np.asarray(biases, dtype=np.float64),
    )


def _normalizer(value: object, input_size: int) -> ZScoreNormalizer | None:
    if value is None:
        return None
    data = _object(value, "normalization")
    parameters = {
        "method": data.get("method"),
        "means": _vector(data.get("means"), input_size, "normalization.means"),
        "standard_deviations": _vector(
            data.get("standard_deviations"), input_size, "normalization.standard_deviations"
        ),
        "zero_std_divisor": _number(
            data.get("zero_std_divisor", 1.0), "normalization.zero_std_divisor"
        ),
    }
    try:
        return ZScoreNormalizer.from_dict(parameters)
    except NormalizationError as error:
        raise PersistenceError(f"normalization: {error}") from error


def _columns(data: dict[str, Any], field: str, size: int) -> tuple[str, ...] | None:
    if field not in data:
        return None
    values = data[field]
    if (
        not isinstance(values, list) or len(values) != size
        or any(not isinstance(value, str) or not value.strip() for value in values)
        or len(set(values)) != size
    ):
        raise PersistenceError(f"network.{field} deve conter {size} nomes distintos.")
    return tuple(values)


def _validate(data: object) -> _SavedNetwork:
    root = _object(data, "Modelo")
    version = root.get("format_version")
    if type(version) is not int or version != FORMAT_VERSION:
        raise PersistenceError(f"Versão de modelo não suportada. Esperado {FORMAT_VERSION}.")

    network = _object(root.get("network"), "network")
    input_size = _positive_int(network.get("input_size"), "network.input_size")
    layer_data = network.get("layers")
    if not isinstance(layer_data, list) or not layer_data:
        raise PersistenceError("network.layers deve ser uma lista com pelo menos uma camada.")
    if "normalization" not in network:
        raise PersistenceError("Campo obrigatório network.normalization ausente.")

    layers = []
    fan_in = input_size
    for index, value in enumerate(layer_data, start=1):
        layer = _layer(value, fan_in, index)
        layers.append(layer)
        fan_in = layer.config.neurons
    return _SavedNetwork(
        input_size, layers, _normalizer(network["normalization"], input_size),
        _columns(network, "input_columns", input_size),
        _columns(network, "output_columns", layers[-1].config.neurons),
    )


def _reject_non_json_number(token: str) -> None:
    raise ValueError(f"Número inválido no JSON: {token}.")


def save_network(network: NeuralNetwork, path: str | Path) -> None:
    """Salva arquitetura, parâmetros e normalização em formato JSON."""
    model_path = Path(path).expanduser()
    temporary_path: Path | None = None
    try:
        data = {
            "format_version": FORMAT_VERSION,
            "network": {
                "input_size": network.input_size,
                "layers": [
                    {
                        "neurons": layer.neuron_count,
                        "activation": layer.activation_name,
                        "weights": [neuron.weights.tolist() for neuron in layer.neurons],
                        "biases": [float(neuron.bias) for neuron in layer.neurons],
                    }
                    for layer in network.layers
                ],
                "normalization": network.normalization,
            },
        }
        if network.input_columns is not None:
            data["network"]["input_columns"] = list(network.input_columns)
        if network.output_columns is not None:
            data["network"]["output_columns"] = list(network.output_columns)
        _validate(data)
        contents = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=model_path.parent, delete=False
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(contents + "\n")
        temporary_path.replace(model_path)
        temporary_path = None
    except (OSError, TypeError, ValueError, OverflowError, PersistenceError, NormalizationError) as error:
        raise PersistenceError(f"Falha ao salvar o modelo em '{model_path}': {error}") from error
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass  # Preserva o erro de gravação original.


def load_network(path: str | Path) -> NeuralNetwork:
    """Carrega e valida uma rede persistida em formato compatível."""
    model_path = Path(path).expanduser()
    try:
        contents = model_path.read_text(encoding="utf-8")
        data = json.loads(contents, parse_constant=_reject_non_json_number)
        model = _validate(data)
        network = NeuralNetwork(model.input_size, [layer.config for layer in model.layers])
        for layer, saved in zip(network.layers, model.layers, strict=True):
            for neuron, weights, bias in zip(
                layer.neurons, saved.weights, saved.biases, strict=True
            ):
                neuron.weights = weights.copy()
                neuron.bias = float(bias)
        network.normalizer = model.normalizer
        network.input_columns = model.input_columns
        network.output_columns = model.output_columns
    except (OSError, ValueError, PersistenceError, RecursionError) as error:
        raise PersistenceError(f"Falha ao carregar o modelo em '{model_path}': {error}") from error
    return network


