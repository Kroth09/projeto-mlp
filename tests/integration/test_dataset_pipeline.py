import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from mlp.config import LayerConfig, load_config
from mlp.data_pipeline import prepare_data
from mlp.network import NeuralNetwork


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_network(config) -> NeuralNetwork:
    return NeuralNetwork(
        config.network.input_size,
        config.network.layers,
        config.network.seed,
    )


def test_complete_pipeline_with_real_dataset() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "example.json", relative_to=PROJECT_ROOT)
    network = build_network(config)

    prepared = prepare_data(config, network)

    assert prepared.dataset.sample_count == 179
    assert prepared.dataset.input_columns == ("sementes_m2", "nitrogenio_kg_ha")
    assert prepared.dataset.output_columns == ("produtividade_kg_ha",)
    assert (
        prepared.split.training.sample_count,
        prepared.split.validation.sample_count,
        prepared.split.test.sample_count,
    ) == (125, 26, 28)
    np.testing.assert_allclose(
        prepared.split.training.inputs.mean(axis=0), [0.0, 0.0], atol=1e-12
    )
    np.testing.assert_allclose(
        prepared.split.training.inputs.std(axis=0), [1.0, 1.0], atol=1e-12
    )
    assert prepared.normalizer is not None
    assert network.normalization == prepared.normalizer.to_dict()
    prediction = network.forward(prepared.split.training.inputs[0])
    assert prediction.shape == (1,)
    assert np.isfinite(prediction).all()


def test_pipeline_without_normalization_keeps_raw_scale() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "example.json", relative_to=PROJECT_ROOT)
    config = replace(
        config,
        normalization=replace(config.normalization, enabled=False),
    )
    network = build_network(config)

    prepared = prepare_data(config, network)

    assert prepared.normalizer is None
    assert network.normalization is None
    assert prepared.split.training.inputs.std(axis=0).min() > 1.0


def test_pipeline_resolves_temp_config_and_preserves_configured_order(
    tmp_path: Path,
) -> None:
    (tmp_path / "data.csv").write_text(
        "y,x2,x1\n3,2,1\n6,5,4\n9,8,7\n12,11,10\n15,14,13\n16,17,18\n",
        encoding="utf-8",
    )
    raw_config = json.loads(
        (PROJECT_ROOT / "configs" / "example.json").read_text(encoding="utf-8")
    )
    raw_config["dataset"].update(
        path="data.csv",
        input_columns=["x1", "x2"],
        output_columns=["y"],
        encoding="utf-8",
    )
    raw_config["network"].update(
        input_size=2,
        layers=[{"neurons": 2, "activation": "tanh"}, {"neurons": 1, "activation": "linear"}],
    )
    raw_config["split"].update(train=50, validation=25, test=25)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(raw_config), encoding="utf-8")
    config = load_config(config_path)

    prepared = prepare_data(config, build_network(config))

    assert prepared.dataset.input_columns == ("x1", "x2")
    np.testing.assert_array_equal(prepared.dataset.inputs[0], [1.0, 2.0])
    assert prepared.split.sample_count == 6

