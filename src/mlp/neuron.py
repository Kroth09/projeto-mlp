"""Entidade que armazena o estado de um neurônio artificial."""

from collections.abc import Sequence

import numpy as np

from mlp.types import FloatArray


class Neuron:
    """Armazena pesos, viés e valores intermediários de um neurônio."""

    def __init__(
        self,
        weights: Sequence[float] | FloatArray,
        bias: float = 0.0,
    ) -> None:
        """Cria um neurônio com parâmetros fornecidos explicitamente."""
        weights_array = np.asarray(weights, dtype=np.float64)
        if weights_array.ndim != 1:
            raise ValueError("Os pesos do neurônio devem formar um vetor.")
        if weights_array.size == 0:
            raise ValueError("O neurônio deve possuir pelo menos um peso.")
        if not np.isfinite(weights_array).all() or not np.isfinite(float(bias)):
            raise ValueError("Pesos e viés devem conter somente valores finitos.")

        self.weights: FloatArray = weights_array.copy()
        self.bias = float(bias)
        self.z: float | None = None
        self.output: float | None = None
        self.delta: float | None = None

    def weighted_sum(self, inputs: FloatArray) -> float:
        """Calcula ``dot(weights, inputs) + bias``."""
        inputs_array = np.asarray(inputs, dtype=np.float64)
        if inputs_array.shape != self.weights.shape:
            raise ValueError(
                "Quantidade de entradas incompatível com os pesos do neurônio."
            )
        if not np.isfinite(inputs_array).all():
            raise ValueError("As entradas devem conter somente valores finitos.")

        result = float(np.dot(self.weights, inputs_array) + self.bias)
        if not np.isfinite(result):
            raise FloatingPointError("Soma ponderada não finita no neurônio.")
        return result

    def forward(self, inputs: FloatArray) -> float:
        """Calcula e armazena a soma ponderada do neurônio."""
        self.z = self.weighted_sum(inputs)
        self.output = None
        self.delta = None
        return self.z

