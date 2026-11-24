"""Inspeciona o CSV fornecido sem alterar o arquivo original."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "raw" / "dataset.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        nargs="?",
        type=Path,
        default=DEFAULT_DATASET,
        help="Caminho do CSV. O padrão é data/raw/dataset.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.dataset, encoding="utf-8-sig")
    numeric = frame.apply(pd.to_numeric, errors="raise")
    finite_counts = {
        column: int(np.isfinite(numeric[column].to_numpy(dtype=float)).sum())
        for column in numeric.columns
    }

    print(f"Arquivo: {args.dataset}")
    print(f"Dimensões: {frame.shape[0]} linhas x {frame.shape[1]} colunas")
    print(f"Colunas: {frame.columns.tolist()}")
    print("\nTipos:")
    print(frame.dtypes.to_string())
    print("\nValores ausentes por coluna:")
    print(frame.isna().sum().to_string())
    print("\nValores finitos por coluna:")
    for column, count in finite_counts.items():
        print(f"{column}: {count}/{len(frame)}")
    print(f"\nLinhas exatamente duplicadas: {int(frame.duplicated().sum())}")
    print("\nEstatísticas:")
    print(numeric.describe().to_string())
    print("\nContagem por combinação de entrada:")
    print(
        numeric.groupby(["sementes_m2", "nitrogenio_kg_ha"])
        .size()
        .rename("observacoes")
        .to_string()
    )


if __name__ == "__main__":
    main()

