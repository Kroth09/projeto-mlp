import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_phase1_data() -> tuple[dict, pd.DataFrame]:
    config = json.loads(
        (PROJECT_ROOT / "configs" / "example.json").read_text(encoding="utf-8")
    )
    dataset = config["dataset"]
    frame = pd.read_csv(
        (PROJECT_ROOT / dataset["path"]).resolve(),
        sep=dataset["delimiter"],
        encoding=dataset["encoding"],
    )
    return config, frame


def test_dataset_shape_and_columns() -> None:
    _, frame = load_phase1_data()
    assert frame.shape == (179, 3)
    assert frame.columns.tolist() == [
        "sementes_m2",
        "nitrogenio_kg_ha",
        "produtividade_kg_ha",
    ]


def test_selected_values_are_complete_numeric_and_finite() -> None:
    config, frame = load_phase1_data()
    dataset = config["dataset"]
    columns = dataset["input_columns"] + dataset["output_columns"]
    numeric = frame[columns].apply(pd.to_numeric, errors="raise")
    assert not numeric.isna().any().any()
    assert np.isfinite(numeric.to_numpy(dtype=float)).all()


def test_network_dimensions_match_dataset() -> None:
    config, _ = load_phase1_data()
    assert config["network"]["input_size"] == len(
        config["dataset"]["input_columns"]
    )
    assert config["network"]["layers"][-1]["neurons"] == len(
        config["dataset"]["output_columns"]
    )


def test_split_percentages_and_expected_counts() -> None:
    config, frame = load_phase1_data()
    split = config["split"]
    percentages = np.array(
        [split["train"], split["validation"], split["test"]], dtype=float
    )
    assert percentages.sum() == 100

    train_count = int(len(frame) * percentages[0] / 100.0)
    validation_count = int(len(frame) * percentages[1] / 100.0)
    test_count = len(frame) - train_count - validation_count
    counts = [train_count, validation_count, test_count]
    assert counts == [125, 26, 28]
    assert sum(counts) == len(frame)

