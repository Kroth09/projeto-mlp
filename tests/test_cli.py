import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mlp.cli import build_parser, main
from mlp.network import NeuralNetwork
from mlp.normalization import ZScoreNormalizer
from mlp.persistence import load_network, save_network
from mlp.trainer import Trainer


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_cli(cwd, *args):
    return subprocess.run(
        [sys.executable, "-m", "mlp", *map(str, args)], cwd=cwd,
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")},
        capture_output=True, text=True, check=False,
    )


@pytest.fixture
def example(tmp_path):
    config = json.loads((PROJECT_ROOT / "configs" / "example.json").read_text(encoding="utf-8"))
    config["dataset"].update(path="data.csv", input_columns=["x2", "x1"], output_columns=["y"])
    config["network"]["layers"] = [
        {"neurons": 2, "activation": "tanh"}, {"neurons": 1, "activation": "linear"},
    ]
    config["training"]["epochs"] = 2
    config["split"].update(train=60, validation=20, test=20)
    (tmp_path / "data.csv").write_text(
        "x1,x2,y,ignorada\n" + "".join(f"{i},{i % 4},{i + 1},extra\n" for i in range(20)),
        encoding="utf-8",
    )
    config_path = tmp_path / "configs" / "custom.json"
    config_path.parent.mkdir()
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path, config


def test_cli_without_command_prints_help(capsys) -> None:
    exit_code = main([])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "MLP configurável" in captured.out
    assert "inspect-dataset" in captured.out


def test_cli_registers_expected_commands() -> None:
    parser = build_parser()
    assert parser.parse_args(["train", "--config", "a.json", "--output", "m.json"]).command == "train"
    assert parser.parse_args(["evaluate", "--config", "a.json", "--model", "m.json"]).subset == "test"
    assert parser.parse_args(["predict", "--model", "m.json", "--values", "2"]).values == [2.0]
    assert parser.parse_args(["inspect-dataset", "--config", "a.json"]).command == "inspect-dataset"


@pytest.mark.parametrize("args", [
    ["inspect-dataset"], ["unknown"], ["train", "--config", "a.json"],
    ["evaluate", "--config", "a.json", "--model", "m.json", "--subset", "other"],
    ["predict", "--model", "m.json", "--values", "not-a-number"],
])
def test_usage_errors_have_exit_code_two(tmp_path, args) -> None:
    result = run_cli(tmp_path, *args)
    assert result.returncode == 2
    assert result.stdout == ""
    assert "usage:" in result.stderr


@pytest.mark.parametrize("final_run", [False, True])
def test_train_cli_exports_history_and_runs_test_only_on_request(
    tmp_path, example, capsys, monkeypatch, final_run
) -> None:
    path, _ = example
    evaluated_sizes = []
    original_evaluate = Trainer.evaluate

    def record_evaluate(self, network, subset):
        evaluated_sizes.append(subset.sample_count)
        return original_evaluate(self, network, subset)

    monkeypatch.setattr(Trainer, "evaluate", record_evaluate)
    monkeypatch.chdir(tmp_path)
    args = [
        "train", "--config", str(path), "--output", "results/model.json",
        "--history", "results/history.csv",
    ]
    if final_run:
        args.append("--final-test")

    assert main(args) == 0
    expected_sizes = [12, 4, 12, 4, 4] if final_run else [12, 4, 12, 4]
    assert evaluated_sizes == expected_sizes
    assert ("MSE final de teste" in capsys.readouterr().out) == final_run
    with (tmp_path / "results" / "history.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ["epoch", "training_mse", "validation_mse"]
        assert [row["epoch"] for row in reader] == ["1", "2"]
    network = load_network(tmp_path / "results" / "model.json")
    assert network.input_columns == ("x2", "x1")
    assert network.output_columns == ("y",)
    assert [layer.activation_name for layer in network.layers] == ["tanh", "linear"]


def test_train_cli_reports_missing_config(capsys, tmp_path) -> None:
    assert main([
        "train", "--config", str(tmp_path / "missing.json"), "--output", str(tmp_path / "model.json"),
    ]) == 1
    assert "Não foi possível ler a configuração" in capsys.readouterr().err


def test_inspect_uses_working_directory_and_reports_column_order(tmp_path, example) -> None:
    path, _ = example
    result = run_cli(tmp_path, "inspect-dataset", "--config", path.relative_to(tmp_path))
    assert result.returncode == 0, result.stderr
    assert "20 linhas x 3 colunas" in result.stdout
    assert "Entradas (2): x2, x1" in result.stdout
    assert "treino=12, validação=4, teste=4" in result.stdout
    assert "Seed da divisão: 42" in result.stdout
    assert "Normalização: zscore" in result.stdout


def test_train_and_evaluate_reproduce_test_without_optional_history(tmp_path, example) -> None:
    path, _ = example
    train = run_cli(tmp_path, "train", "--config", path, "--output", "out/model.json", "--final-test")
    assert train.returncode == 0, train.stderr
    assert "Época 1/2" in train.stdout and "Época 2/2" in train.stdout
    assert list((tmp_path / "out").iterdir()) == [tmp_path / "out" / "model.json"]
    assert not (tmp_path / "outputs" / "training_history.csv").exists()
    mse = next(line for line in train.stdout.splitlines() if line.startswith("MSE final de teste:"))

    evaluate = run_cli(tmp_path, "evaluate", "--config", path, "--model", "out/model.json")
    assert evaluate.returncode == 0, evaluate.stderr
    assert evaluate.stdout.strip() == mse.replace("MSE final de teste", "MSE de teste")


def test_evaluate_all_uses_model_normalizer_and_accepts_override(tmp_path, example) -> None:
    path, config = example
    config["normalization"]["enabled"] = False
    config["split"]["seed"] = None
    path.write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "external.csv").write_text("x1,x2,y\n25,14,5\n30,20,9\n", encoding="utf-8")
    network = NeuralNetwork(2, [{"neurons": 1, "activation": "linear"}], seed=42)
    neuron = network.layers[0].neurons[0]
    neuron.weights[:] = [2.0, -1.0]
    neuron.bias = 3.0
    network.normalizer = ZScoreNormalizer.from_dict({
        "method": "zscore", "means": [10, 20], "standard_deviations": [2, 5],
    })
    network.input_columns = ("x2", "x1")
    save_network(network, tmp_path / "model.json")
    args = ["evaluate", "--config", path, "--model", "model.json", "--dataset", "external.csv"]
    evaluate = run_cli(tmp_path, *args, "--subset", "all")
    assert evaluate.returncode == 0, evaluate.stderr
    assert evaluate.stdout.strip() == "MSE no CSV completo: 2.500000 (2 exemplos)"

    no_seed = run_cli(tmp_path, *args)
    assert no_seed.returncode == 1
    assert "split.seed" in no_seed.stderr

    config["dataset"]["input_columns"].reverse()
    path.write_text(json.dumps(config), encoding="utf-8")
    wrong_order = run_cli(tmp_path, *args, "--subset", "all")
    assert wrong_order.returncode == 1
    assert "ordem registrada" in wrong_order.stderr


def test_predict_cli_returns_json_and_reports_dimension_error(tmp_path) -> None:
    network = NeuralNetwork(2, [{"neurons": 1, "activation": "linear"}], seed=42)
    neuron = network.layers[0].neurons[0]
    neuron.weights[:] = [-0.2, 0.0]
    neuron.bias = 0.0
    network.normalizer = ZScoreNormalizer(
        means=np.array([10.0, 20.0]), standard_deviations=np.array([2.0, 5.0]),
    )
    save_network(network, tmp_path / "model.json")
    args = ["predict", "--model", "model.json", "--values"]
    predict = run_cli(tmp_path, *args, "14", "25")
    assert predict.returncode == 0, predict.stderr
    assert json.loads(predict.stdout) == {"prediction": [-0.4]}

    wrong_dimension = run_cli(tmp_path, *args, "14")
    assert wrong_dimension.returncode == 1
    assert wrong_dimension.stdout == ""
    assert "2 entradas" in wrong_dimension.stderr


def test_runtime_errors_report_stderr_without_traceback(tmp_path, example) -> None:
    path, config = example
    (tmp_path / "broken.json").write_text("{", encoding="utf-8")
    (tmp_path / "bad-utf8.json").write_bytes(b"\xff")
    cases = [
        (["inspect-dataset", "--config", path, "--dataset", "missing.csv"], "Arquivo não encontrado"),
        (["inspect-dataset", "--config", "broken.json"], "JSON inválido"),
        (["inspect-dataset", "--config", "bad-utf8.json"], "Não foi possível ler"),
        (["predict", "--model", "missing.json", "--values", "1"], "Falha ao carregar"),
        (["predict", "--model", "broken.json", "--values", "1"], "Falha ao carregar"),
    ]
    for args, message in cases:
        result = run_cli(tmp_path, *args)
        assert result.returncode == 1
        assert result.stdout == ""
        assert message in result.stderr and "Traceback" not in result.stderr

    config["dataset"]["delimiter"] = "\n"
    path.write_text(json.dumps(config), encoding="utf-8")
    invalid_delimiter = run_cli(tmp_path, "inspect-dataset", "--config", path)
    assert invalid_delimiter.returncode == 1
    assert "Não foi possível ler o CSV" in invalid_delimiter.stderr


def test_train_rejects_conflicting_destinations(tmp_path, example, capsys, monkeypatch) -> None:
    path, _ = example
    monkeypatch.chdir(tmp_path)
    originals = [(path, path.read_bytes()), (tmp_path / "data.csv", (tmp_path / "data.csv").read_bytes())]
    args = ["train", "--config", str(path), "--output"]
    assert main([*args, "data.csv"]) == 1
    assert main([*args, "model.json", "--history", str(path)]) == 1
    assert main([*args, "model.json", "--history", "model.json"]) == 1
    assert all(file.read_bytes() == contents for file, contents in originals)
    assert not (tmp_path / "model.json").exists()
    assert "destinos" in capsys.readouterr().err

