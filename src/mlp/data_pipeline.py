"""Integra configuração, dataset, divisão, normalização e rede."""

from dataclasses import dataclass

from mlp.config import ApplicationConfig, DatasetConfig, SplitConfig
from mlp.dataset import (
    DatasetSplit,
    SupervisedDataset,
    load_csv_dataset,
    split_dataset,
    validate_dataset_compatibility,
)
from mlp.network import NeuralNetwork
from mlp.normalization import ZScoreNormalizer, normalize_split


@dataclass(frozen=True)
class PreparedData:
    """Resultado completo da preparação de um dataset."""

    dataset: SupervisedDataset
    split: DatasetSplit
    normalizer: ZScoreNormalizer | None


def load_configured_dataset(config: DatasetConfig) -> SupervisedDataset:
    """Lê os dados na ordem das colunas da configuração."""
    return load_csv_dataset(
        path=config.path,
        input_columns=config.input_columns,
        output_columns=config.output_columns,
        delimiter=config.delimiter,
        encoding=config.encoding,
    )


def split_configured_dataset(dataset: SupervisedDataset, config: SplitConfig) -> DatasetSplit:
    """Divide os dados com os percentuais e a seed da configuração."""
    return split_dataset(
        dataset=dataset,
        train_percentage=config.train,
        validation_percentage=config.validation,
        test_percentage=config.test,
        seed=config.seed,
    )


def prepare_data(config: ApplicationConfig, network: NeuralNetwork) -> PreparedData:
    """Executa o fluxo obrigatório sem normalizar as saídas."""
    dataset = load_configured_dataset(config.dataset)
    validate_dataset_compatibility(dataset, network)
    split = split_configured_dataset(dataset, config.split)

    normalizer: ZScoreNormalizer | None = None
    if config.normalization.enabled:
        split, normalizer = normalize_split(
            split,
            ZScoreNormalizer(
                zero_std_divisor=config.normalization.zero_std_divisor
            ),
        )
    network.normalizer = normalizer
    network.input_columns = dataset.input_columns
    network.output_columns = dataset.output_columns
    return PreparedData(dataset, split, normalizer)

