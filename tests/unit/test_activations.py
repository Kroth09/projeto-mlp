import math

import pytest

from mlp.activations import (
    get_activation,
    linear,
    linear_derivative,
    relu,
    relu_derivative,
    sigmoid,
    sigmoid_derivative,
    tanh,
    tanh_derivative,
)


@pytest.mark.parametrize(
    ("function", "value", "expected"),
    [
        (linear, -2.0, -2.0),
        (linear, 0.0, 0.0),
        (sigmoid, 0.0, 0.5),
        (tanh, 0.0, 0.0),
        (relu, -2.0, 0.0),
        (relu, 3.0, 3.0),
    ],
)
def test_activation_known_values(function, value: float, expected: float) -> None:
    assert function(value) == pytest.approx(expected)


def test_derivative_known_values() -> None:
    assert linear_derivative(-10.0) == 1.0
    assert sigmoid_derivative(0.0) == pytest.approx(0.25)
    assert tanh_derivative(0.0) == pytest.approx(1.0)
    assert relu_derivative(-1.0) == 0.0
    assert relu_derivative(0.0) == 0.0
    assert relu_derivative(2.0) == 1.0


def test_sigmoid_is_stable_for_large_magnitudes() -> None:
    positive = sigmoid(1000.0)
    negative = sigmoid(-1000.0)

    assert math.isfinite(positive)
    assert math.isfinite(negative)
    assert positive == pytest.approx(1.0)
    assert negative == pytest.approx(0.0)


def test_activation_registry_is_case_insensitive() -> None:
    activation = get_activation(" TaNh ")
    assert activation.function(0.25) == pytest.approx(tanh(0.25))
    assert activation.derivative(0.25) == pytest.approx(tanh_derivative(0.25))


def test_unknown_activation_is_rejected() -> None:
    with pytest.raises(ValueError, match="desconhecida"):
        get_activation("softmax")


@pytest.mark.parametrize("invalid_name", [None, 123, ""])
def test_invalid_activation_name_type_is_rejected(invalid_name) -> None:
    with pytest.raises(ValueError, match="string não vazia"):
        get_activation(invalid_name)

