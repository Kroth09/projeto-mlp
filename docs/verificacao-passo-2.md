# Verificação do passo 2

Verificação executada no ambiente virtual do projeto.

## Versão do Python

```text
Python 3.12.3
```

## Pacotes instalados

```text
configurable-mlp 0.1.0 (instalação editável)
iniconfig        2.3.1
numpy            2.5.5
packaging        26.4
pandas           3.1.1
pip              26.3
pluggy           1.7.0
Pygments         2.21.0
pytest           9.2.0
python-dateutil  4.0.0
setuptools       84.2.1
tzdata           2026.6
```

## Testes

Comando:

```bash
python -m pytest
```

Resultado:

```text
...................                                                      [100%]
19 passed in 0.31s
```

## Ajuda da CLI

Os comandos `python -m mlp --help` e `mlp --help` produziram a mesma saída e terminaram sem traceback:

```text
usage: mlp [-h] COMANDO ...

MLP configurável implementada sem frameworks de redes neurais.

positional arguments:
  COMANDO
    train          Treina uma nova rede.
    evaluate       Avalia uma rede salva.
    predict        Executa uma previsão.
    inspect-dataset
                   Mostra informações básicas sobre um CSV.
```

## Árvore do projeto

```text
projeto-mlp/
├── configs/
│   └── example.json
├── data/
│   ├── processed/
│   │   └── .gitkeep
│   └── raw/
│       └── dataset.csv
├── docs/
│   ├── decisoes.md
│   └── verificacao-passo-2.md
├── outputs/
│   └── .gitkeep
├── scripts/
│   ├── inspect_dataset.py
│   └── validate_phase1.py
├── src/mlp/
│   ├── __init__.py
│   ├── __main__.py
│   ├── activations.py
│   ├── cli.py
│   ├── config.py
│   ├── dataset.py
│   ├── exceptions.py
│   ├── layer.py
│   ├── network.py
│   ├── neuron.py
│   ├── normalization.py
│   ├── persistence.py
│   ├── trainer.py
│   └── types.py
├── tests/
│   ├── test_cli.py
│   ├── test_config.py
│   ├── test_environment.py
│   ├── test_imports.py
│   └── test_phase1.py
├── .gitignore
├── pyproject.toml
├── README.md
└── requirements.txt
```

O ambiente virtual, caches e metadados gerados pela instalação foram omitidos da árvore.

