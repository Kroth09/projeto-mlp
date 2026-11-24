# Verificação da CLI e do fluxo final

## Execução com o dataset real

Comandos executados a partir da raiz do projeto com Python 3.12 e o ambiente instalado.

```bash
mlp --help
python -m mlp inspect-dataset --config configs/example.json
mlp train --config configs/example.json --output outputs/model.json \
  --history outputs/training_history.csv --final-test
python -m mlp evaluate --config configs/example.json --model outputs/model.json
mlp evaluate --config configs/example.json --model outputs/model.json --subset all
mlp predict --model outputs/model.json --values 450 120
```

O `inspect-dataset` reconheceu 179 linhas, duas entradas e uma saída. A divisão com seed 42 gerou 125 exemplos de treino, 26 de validação e 28 de teste. O treino exibiu 196 linhas de métricas por época e exportou o modelo e o histórico. O JSON registra a ordem das colunas de entrada e saída. A avaliação confere essa ordem quando o modelo contém os metadados.

| Medição | Exemplos | Resultado |
|---|---:|---:|
| MSE de treino na última época | 125 | 52.401,788317 |
| MSE de validação na última época | 26 | 55.820,740259 |
| MSE de teste calculado pelo `train --final-test` | 28 | 58.473,198175 |
| MSE de teste calculado pelo `evaluate` com o modelo recarregado | 28 | 58.473,198175 |
| MSE no CSV completo | 179 | 53.848,113605 |

O vetor bruto `[450, 120]`, na ordem `sementes_m2` e `nitrogenio_kg_ha`, produz `{"prediction": [2271.021133610547]}` em kg/ha. O histórico coincide com o da fase 6 e a cópia do dataset coincide com o arquivo original.

## Casos de teste

A suíte aprovou 180 testes. Os testes da CLI verificam argumentos obrigatórios, separação de `stdout` e `stderr`, códigos de saída, caminhos relativos à pasta de execução, proteção dos arquivos de entrada, modelo carregável e reprodução do MSE reservado. A avaliação de um CSV externo pequeno confirmou que o Z-score vem do modelo salvo. A CLI também rejeita mudanças na ordem das colunas registradas.

As verificações do núcleo e dos dados foram repetidas. O maior erro relativo na checagem de gradientes foi `2,913141949890e-12`. O teste sintético de normalização confirmou o ajuste das estatísticas exclusivamente no treino. A suíte também confere a reprodutibilidade e a equivalência das previsões após carregar o modelo.
