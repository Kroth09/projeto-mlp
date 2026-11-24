import copy
import json
from pathlib import Path

import pytest

from mlp.config import ApplicationConfig, load_config
from mlp.exceptions import ConfigurationError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = PROJECT_ROOT / "configs" / "example.json"


def read_valid_data() -> dict:
    return json.loads(EXAMPLE_CONFIG.read_text(encoding="utf-8"))


def write_config(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_loads_valid_config() -> None:
    config = load_config(EXAMPLE_CONFIG, relative_to=PROJECT_ROOT)

    assert isinstance(config, ApplicationConfig)
    assert config.network.input_size == 2
    assert [layer.neurons for layer in config.network.layers] == [8, 4, 1]
    assert config.dataset.input_columns == ("sementes_m2", "nitrogenio_kg_ha")
    assert config.dataset.output_columns == ("produtividade_kg_ha",)
    assert config.dataset.path == PROJECT_ROOT / "data" / "raw" / "dataset.csv"
    assert config.output.model_path == PROJECT_ROOT / "outputs" / "model.json"


def test_relative_paths_do_not_depend_on_current_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = write_config(tmp_path, read_valid_data())
    working_directory = tmp_path / "work"
    working_directory.mkdir()
    monkeypatch.chdir(working_directory)

    config = load_config(config_path)

    assert config.dataset.path == tmp_path / "data" / "raw" / "dataset.csv"
    assert config.output.history_path == (
        tmp_path / "outputs" / "training_history.csv"
    )


def test_rejects_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="JSON inválido"):
        load_config(path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda data: data["network"].update(input_size=0), "inteiro positivo"),
        (lambda data: data["network"].update(layers=[]), "lista não vazia"),
        (
            lambda data: data["network"]["layers"][0].update(activation="softmax"),
            "activation desconhecida",
        ),
        (lambda data: data["split"].update(test=20), "totalizar 100"),
        (
            lambda data: data["split"].update(rounding="largest_remainder"),
            "remainder_to_test",
        ),
    ],
)
def test_rejects_invalid_structural_values(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    data = copy.deepcopy(read_valid_data())
    mutation(data)

    with pytest.raises(ConfigurationError, match=message):
        load_config(write_config(tmp_path, data))


def test_rejects_input_count_incompatible_with_dataset(tmp_path: Path) -> None:
    data = read_valid_data()
    data["network"]["input_size"] = 3

    with pytest.raises(ConfigurationError, match="quantidade de colunas de entrada"):
        load_config(write_config(tmp_path, data))


def test_rejects_present_but_empty_output_object(tmp_path: Path) -> None:
    data = read_valid_data()
    data["output"] = {}

    with pytest.raises(ConfigurationError, match="output.model_path"):
        load_config(write_config(tmp_path, data))


def test_missing_output_object_uses_paths_relative_to_config(tmp_path: Path) -> None:
    data = read_valid_data()
    del data["output"]

    config = load_config(write_config(tmp_path, data))

    assert config.output.model_path == tmp_path / "outputs" / "model.json"
    assert config.output.history_path == tmp_path / "outputs" / "training_history.csv"

