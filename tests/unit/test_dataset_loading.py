from pathlib import Path

import numpy as np
import pytest

from mlp.dataset import load_csv_dataset
from mlp.exceptions import DatasetError


def write_csv(tmp_path: Path, content: str, name: str = "dataset.csv") -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_loads_float64_matrices_in_configured_column_order(tmp_path: Path) -> None:
    path = write_csv(tmp_path, "y,x2,x1\n30,2,1\n60,5,4\n")

    dataset = load_csv_dataset(path, ["x1", "x2"], ["y"])

    np.testing.assert_array_equal(dataset.inputs, [[1.0, 2.0], [4.0, 5.0]])
    np.testing.assert_array_equal(dataset.expected_outputs, [[30.0], [60.0]])
    assert dataset.inputs.dtype == np.float64
    assert dataset.expected_outputs.dtype == np.float64
    assert dataset.inputs.ndim == dataset.expected_outputs.ndim == 2
    assert dataset.input_columns == ("x1", "x2")
    assert dataset.output_columns == ("y",)
    assert not dataset.inputs.flags.writeable


def test_accepts_multiple_outputs_and_custom_delimiter(tmp_path: Path) -> None:
    path = write_csv(tmp_path, "x;y2;y1\n1;20;10\n2;40;30\n")

    dataset = load_csv_dataset(path, ["x"], ["y1", "y2"], delimiter=";")

    np.testing.assert_array_equal(dataset.expected_outputs, [[10.0, 20.0], [30.0, 40.0]])


def test_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="Arquivo não encontrado"):
        load_csv_dataset(tmp_path / "missing.csv", ["x"], ["y"])


def test_rejects_directory_path(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="não é um arquivo"):
        load_csv_dataset(tmp_path, ["x"], ["y"])


def test_rejects_empty_csv(tmp_path: Path) -> None:
    path = write_csv(tmp_path, "")
    with pytest.raises(DatasetError, match="vazio"):
        load_csv_dataset(path, ["x"], ["y"])


def test_rejects_header_without_examples(tmp_path: Path) -> None:
    path = write_csv(tmp_path, "x,y\n")
    with pytest.raises(DatasetError, match="não possui exemplos"):
        load_csv_dataset(path, ["x"], ["y"])


def test_rejects_missing_column(tmp_path: Path) -> None:
    path = write_csv(tmp_path, "x,y\n1,2\n")
    with pytest.raises(DatasetError, match="Colunas não encontradas: z"):
        load_csv_dataset(path, ["x", "z"], ["y"])


def test_rejects_text_in_selected_columns(tmp_path: Path) -> None:
    path = write_csv(tmp_path, "x,y\ntexto,2\n")
    with pytest.raises(DatasetError, match="somente valores numéricos"):
        load_csv_dataset(path, ["x"], ["y"])


@pytest.mark.parametrize("invalid_value", ["", "NaN", "inf", "-inf"])
def test_rejects_missing_or_non_finite_values(
    tmp_path: Path,
    invalid_value: str,
) -> None:
    path = write_csv(tmp_path, f"x,y\n{invalid_value},2\n")
    with pytest.raises(DatasetError, match="NaN|infinitos"):
        load_csv_dataset(path, ["x"], ["y"])


def test_rejects_duplicate_or_overlapping_column_names(tmp_path: Path) -> None:
    path = write_csv(tmp_path, "x,y\n1,2\n")
    with pytest.raises(DatasetError, match="duplicados"):
        load_csv_dataset(path, ["x", "x"], ["y"])
    with pytest.raises(DatasetError, match="entrada e saída"):
        load_csv_dataset(path, ["x"], ["x"])


def test_rejects_unknown_encoding_with_domain_error(tmp_path: Path) -> None:
    path = write_csv(tmp_path, "x,y\n1,2\n")
    with pytest.raises(DatasetError, match="Não foi possível ler"):
        load_csv_dataset(path, ["x"], ["y"], encoding="encoding-inexistente")

