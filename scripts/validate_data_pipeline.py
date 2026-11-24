"""Gera evidências reproduzíveis da preparação do dataset real."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from mlp.config import load_config
from mlp.data_pipeline import prepare_data
from mlp.dataset import DatasetSplit, DatasetSubset, load_csv_dataset, split_dataset
from mlp.network import NeuralNetwork
from mlp.normalization import ZScoreNormalizer, normalize_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "example.json"


def sha256(path: Path) -> str:
    """Calcula SHA-256 sem alterar o arquivo."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    config = load_config(CONFIG_PATH, relative_to=PROJECT_ROOT)
    network = NeuralNetwork(
        config.network.input_size,
        config.network.layers,
        config.network.seed,
    )
    raw_dataset = load_csv_dataset(
        config.dataset.path,
        config.dataset.input_columns,
        config.dataset.output_columns,
        config.dataset.delimiter,
        config.dataset.encoding,
    )
    raw_split = split_dataset(
        raw_dataset,
        config.split.train,
        config.split.validation,
        config.split.test,
        config.split.seed,
    )
    prepared = prepare_data(config, network)
    normalizer = prepared.normalizer
    if normalizer is None:
        raise RuntimeError("A configuração de aceitação deve habilitar Z-score.")

    outputs_unchanged = all(
        np.array_equal(raw.expected_outputs, normalized.expected_outputs)
        for raw, normalized in zip(
            (raw_split.training, raw_split.validation, raw_split.test),
            (
                prepared.split.training,
                prepared.split.validation,
                prepared.split.test,
            ),
            strict=True,
        )
    )
    restored = ZScoreNormalizer.from_dict(normalizer.to_dict())
    round_trip_difference = np.max(
        np.abs(
            restored.transform(raw_split.test.inputs)
            - prepared.split.test.inputs
        )
    )

    leak_split = DatasetSplit(
        training=DatasetSubset(np.array([[0.0], [2.0]]), np.array([[0.0], [1.0]])),
        validation=DatasetSubset(np.array([[100.0]]), np.array([[2.0]])),
        test=DatasetSubset(np.array([[200.0]]), np.array([[3.0]])),
        seed=42,
    )
    _, leak_normalizer = normalize_split(leak_split)

    prediction = network.forward(prepared.split.training.inputs[0])
    print(f"Dataset: {config.dataset.path}")
    print(f"SHA-256: {sha256(config.dataset.path)}")
    print(f"Exemplos: {prepared.dataset.sample_count}")
    print(f"Entradas: {list(prepared.dataset.input_columns)}")
    print(f"Saídas: {list(prepared.dataset.output_columns)}")
    print(
        "Divisão: "
        f"treino={prepared.split.training.sample_count}, "
        f"validação={prepared.split.validation.sample_count}, "
        f"teste={prepared.split.test.sample_count}, "
        f"seed={prepared.split.seed}"
    )
    print(f"Médias do treino: {normalizer.means.tolist()}")
    print(f"Desvios do treino: {normalizer.standard_deviations.tolist()}")
    print(
        "Média normalizada do treino: "
        f"{prepared.split.training.inputs.mean(axis=0).tolist()}"
    )
    print(
        "Desvio normalizado do treino: "
        f"{prepared.split.training.inputs.std(axis=0).tolist()}"
    )
    print(
        "Média normalizada da validação: "
        f"{prepared.split.validation.inputs.mean(axis=0).tolist()}"
    )
    print(
        "Média normalizada do teste: "
        f"{prepared.split.test.inputs.mean(axis=0).tolist()}"
    )
    print(f"Saídas preservadas: {outputs_unchanged}")
    print(f"Diferença após serialização: {round_trip_difference:.12e}")
    print(
        "Teste sintético sem vazamento: "
        f"média={leak_normalizer.means.tolist()}, "
        f"desvio={leak_normalizer.standard_deviations.tolist()}"
    )
    print(f"Forward com entrada preparada: shape={prediction.shape}")


if __name__ == "__main__":
    main()

