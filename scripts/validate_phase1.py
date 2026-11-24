"""Valida configuração, dataset, divisão e normalização sem treinar a rede."""

from pathlib import Path

from mlp.config import load_config
from mlp.data_pipeline import prepare_data
from mlp.network import NeuralNetwork


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "example.json"


def main() -> None:
    config = load_config(CONFIG_PATH, relative_to=PROJECT_ROOT)
    network = NeuralNetwork(
        config.network.input_size,
        config.network.layers,
        config.network.seed,
    )
    prepared = prepare_data(config, network)
    split = prepared.split

    print("Preparação do dataset validada.")
    print(f"Registros: {prepared.dataset.sample_count}")
    print(f"Entradas: {list(prepared.dataset.input_columns)}")
    print(f"Saídas: {list(prepared.dataset.output_columns)}")
    print(
        "Divisão: "
        f"treino={split.training.sample_count}, "
        f"validação={split.validation.sample_count}, "
        f"teste={split.test.sample_count}"
    )
    if prepared.normalizer is None:
        print("Normalização: desabilitada")
        return

    print("Normalização: Z-score ajustado somente com o treino")
    print(
        "Médias do treino: "
        f"{dict(zip(prepared.dataset.input_columns, prepared.normalizer.means.tolist(), strict=True))}"
    )
    print(
        "Desvios do treino: "
        f"{dict(zip(prepared.dataset.input_columns, prepared.normalizer.standard_deviations.tolist(), strict=True))}"
    )
    print(
        "Média normalizada do treino: "
        f"{split.training.inputs.mean(axis=0).tolist()}"
    )
    print(
        "Desvio normalizado do treino: "
        f"{split.training.inputs.std(axis=0).tolist()}"
    )


if __name__ == "__main__":
    main()

