# Verificação do passo 6

## Formato do modelo

O JSON em UTF-8 usa `format_version` igual a `1`. O objeto `network` contém `input_size`, a lista ordenada de camadas e o campo `normalization`. Cada camada registra `neurons`, `activation`, `weights` e `biases`. Quando habilitado, o Z-score guarda as médias e os divisores por entrada. Com a normalização desativada, o campo recebe `null`.

O carregamento valida a versão, os campos obrigatórios, a ativação e as dimensões de cada camada antes de criar a rede. Pesos, vieses e parâmetros de normalização devem ser números finitos. A gravação valida o conteúdo e substitui o arquivo de destino após terminar a escrita.

## Cálculo manual

A rede de referência usa duas entradas, dois neurônios ReLU e uma saída Linear. Com `[14, 25]`, o Z-score produz `[2, 1]`, a camada oculta produz `[0,25; 2]` e a saída esperada é `[-0,4]`. O teste confirma esse valor antes de salvar e após carregar, além de comparar os parâmetros de todas as camadas.

## Modelo treinado

O arquivo [outputs/model.json](../outputs/model.json) foi gerado repetindo o treino da fase 5 com o JSON de exemplo. As 196 épocas reproduziram MSE de treino de 52.401,788317 e MSE de validação de 55.820,740259. A ordem das entradas é `sementes_m2`, `nitrogenio_kg_ha`.

| Sementes por m² | Nitrogênio (kg/ha) | Previsão antes de salvar (kg/ha) | Previsão após carregar (kg/ha) |
|---:|---:|---:|---:|
| 150 | 0 | 2.690,549603 | 2.690,549603 |
| 325 | 37,5 | 2.690,549603 | 2.690,549603 |
| 450 | 120 | 2.271,021134 | 2.271,021134 |

A diferença máxima entre as previsões antes e depois do carregamento foi exatamente zero. As três previsões também coincidiram com o `forward` aplicado aos vetores transformados pelo normalizador original.

## Testes automatizados

`python -m pytest` aprovou 170 testes. Os cenários desta fase incluem o cálculo manual, a reconstrução de todos os parâmetros, modelos com e sem Z-score, coluna constante, entradas inválidas, arquivos ausentes ou corrompidos, versão incompatível e preservação do arquivo anterior quando uma tentativa de gravação falha na validação.
