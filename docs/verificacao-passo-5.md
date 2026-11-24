# Verificação do passo 5

## Configuração do experimento

O dataset contém 179 exemplos. A divisão com seed 42 produziu 125 exemplos de treino, 26 de validação e 28 de teste. As entradas `sementes_m2` e `nitrogenio_kg_ha` passaram pelo Z-score ajustado no treino. O alvo `produtividade_kg_ha` foi mantido em kg/ha, então a unidade do MSE é (kg/ha)².

A arquitetura foi `2 → 8 Tanh → 4 Tanh → 1 Linear`. As seeds de inicialização, divisão e embaralhamento foram todas 42. O treino embaralhou os índices a cada época e executou uma atualização SGD por exemplo.

O teste inicial de cinco épocas produziu cinco linhas com erros finitos. Depois, foram comparadas três taxas de aprendizado com até 300 épocas por taxa. Cada tentativa começou com os mesmos pesos iniciais e usou os mesmos subconjuntos.

| Taxa de aprendizado | Época com menor MSE de validação | Menor MSE de validação |
|---:|---:|---:|
| 0,01 | 196 | 55.820,74 |
| 0,003 | 86 | 61.743,34 |
| 0,001 | 212 | 80.298,22 |

A menor perda de validação definiu a configuração final em 196 épocas e taxa 0,01. As tentativas dessa comparação mediram treino e validação. Uma execução preliminar da versão anterior da CLI já havia calculado o teste com 1.000 épocas. O resultado preliminar ficou fora da comparação de hiperparâmetros.

## Execução final

O comando abaixo registra a interface disponível na fase 5. As opções atuais estão na [seção CLI do README](../README.md#cli).

```bash
python -m mlp train --config configs/example.json --final-run
```

| Subconjunto | MSE na época 1 | MSE na época 196 |
|---|---:|---:|
| Treino | 102.591,40 | 52.401,79 |
| Validação | 99.045,25 | 55.820,74 |
| Teste | — | 58.473,20 |

A execução fez 24.500 atualizações SGD e levou cerca de 12 segundos no ambiente da verificação. O histórico tem 196 linhas de dados, numeradas de 1 a 196, com os dois MSEs medidos após as atualizações de cada época. O MSE de teste foi calculado uma vez nessa execução, após a última época.

Histórico completo em [outputs/training_history.csv](../outputs/training_history.csv).

## Testes automatizados

`python -m pytest` aprovou 134 testes. O teste manual do guia confirmou peso e viés finais de `0,12` e MSEs de treino, validação e teste de `0,3176`, `0,0676` e `0,4096`, respectivamente. Os testes também conferem o número e a ordem das atualizações, o pareamento de cada exemplo, a reprodutibilidade com seed, a avaliação com múltiplas saídas, o isolamento de validação e teste e a rejeição de entradas inválidas.
