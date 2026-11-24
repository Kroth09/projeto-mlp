# Verificação do passo 3

## Testes automatizados

Comando:

```bash
python -m pytest
```

Resultado:

```text
.....................................................................    [100%]
69 passed in 4.10s
```

Os testes cobrem:

- ativações e derivadas, incluindo Sigmoid em `±1000`;
- Xavier, He, vieses nulos e reprodutibilidade;
- soma ponderada do neurônio;
- forward de camada e de arquiteturas dinâmicas;
- MSE de vetor e dataset;
- fator `2 / m` com múltiplas saídas;
- deltas ocultos usando os pesos do forward;
- atualização manual de peso e viés;
- erros de dimensão, taxa inválida e backward sem forward;
- gradient check por diferenças centrais;
- aprendizado e reprodutibilidade do XOR.
- resolução de caminhos relativa ao arquivo de configuração;
- rejeição de blocos `output` presentes e incompletos;
- mensagens claras para tipos inválidos nas ativações e inicializações.

## Verificação numérica

Comando:

```bash
python scripts/validate_core.py
```

Resultado:

```text
Forward manual 2 -> 2 -> 1: 0.310000000000
Atualização manual 1 -> 1: peso=0.860000000000, viés=0.280000000000
Maior erro relativo do gradient check: 2.913141949890e-12
XOR MSE inicial: 0.299795563895
XOR MSE final: 0.000107173992
XOR previsões: [0.009592933707799473, 0.9889461763828762, 0.9912884188273249, 0.011772549824805308]
```

## Gradient check

A rede verificada usa duas entradas, dois neurônios Tanh e uma saída Sigmoid, com pesos fixos. Cada peso e viés é perturbado por `epsilon = 1e-5`. O gradiente numérico usa diferenças centrais e é comparado ao gradiente analítico.

O maior erro relativo foi `2,913141949890e-12`, abaixo do limite `1e-5`.

## XOR

Configuração:

```text
arquitetura: 2 -> 4 Tanh -> 1 Sigmoid
learning_rate: 0.1
epochs: 10000
seed de inicialização: 42
seed da ordem dos exemplos: 42
estratégia: SGD, uma atualização por exemplo
```

O MSE caiu de `0,299795563895` para `0,000107173992`. As quatro previsões atendem aos limites menores que `0,2` para a classe zero e maiores que `0,8` para a classe um.

