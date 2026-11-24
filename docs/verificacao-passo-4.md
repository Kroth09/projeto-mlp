# Verificação do passo 4

## Resultado automatizado

Comandos executados no ambiente virtual do projeto:

```bash
python -m pytest
python scripts/validate_phase1.py
python scripts/validate_core.py
python scripts/validate_data_pipeline.py
python -m pip check
python -m compileall -q src scripts tests
```

Resultado da suíte:

```text
118 tests collected
118 testes aprovados
No broken requirements found.
```

## Dataset real

```text
SHA-256: 62ab886966a86e765be07f4233d0b45584e49eee02d49877ee7c5c21ed6e434e
Exemplos: 179
Entradas: [sementes_m2, nitrogenio_kg_ha]
Saída: [produtividade_kg_ha]
Divisão: treino=125, validação=26, teste=28, seed=42
```

As contagens seguem a regra definida para este trabalho: treino e validação são truncados e o teste recebe o restante. A permutação usa uma única seed e preserva o pareamento entre entradas e saídas.

## Z-score sem vazamento

Parâmetros obtidos somente nas 125 entradas de treino:

| Entrada | Média | Desvio-padrão populacional |
|---|---:|---:|
| `sementes_m2` | 305,2000 | 106,1741964886 |
| `nitrogenio_kg_ha` | 62,8800 | 42,8357981133 |

Depois da transformação, as médias do treino ficaram numericamente em zero e os desvios em um:

```text
Média normalizada do treino: [8.881784197001253e-17, -3.730349362740526e-17]
Desvio normalizado do treino: [0.9999999999999992, 1.0]
```

Um teste sintético separado usa treino `[0, 2]`, validação `[100]` e teste `[200]`. O normalizador mantém média `[1]` e desvio `[1]`, provando que validação e teste não participam do ajuste. As saídas são comparadas antes e depois e permanecem idênticas.

## Serialização e integração

Os parâmetros do Z-score são armazenados no objeto da rede. A reconstrução a partir do dicionário serializado produziu diferença máxima `0.0` na transformação. Uma entrada preparada pelo pipeline também percorreu a rede configurada e gerou saída com formato `(1,)`.

## Casos cobertos

- arquivo inexistente, diretório, CSV vazio e somente cabeçalho;
- separador, codificação, colunas ausentes, duplicadas ou sobrepostas;
- texto, `NaN` e infinito em colunas selecionadas;
- incompatibilidade entre dimensões do dataset e da rede;
- percentuais, seed, arredondamento e subconjuntos vazios;
- reprodutibilidade, ausência de perda de linhas e preservação de pares;
- coluna constante, divisor configurável e parâmetros inválidos;
- transformação antes do ajuste e incompatibilidade de atributos;
- serialização do normalizador e pipeline com normalização desativada.

## Critérios de aceite

- [x] CSV real carregado por nomes de colunas e convertido para `float64`.
- [x] Matrizes de entrada e saída mantidas bidimensionais.
- [x] Divisão reproduzível em treino, validação e teste.
- [x] Todas as 179 linhas usadas exatamente uma vez.
- [x] Z-score ajustado somente no treino com `ddof=0`.
- [x] Colunas constantes tratadas sem divisão por zero.
- [x] Saídas preservadas na escala original.
- [x] Parâmetros do normalizador serializáveis e ligados à rede.
- [x] Pipeline integrado validado com o dataset real.
- [x] Suíte completa com 118 testes aprovada.
