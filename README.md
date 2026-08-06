# Detecção de Fraude em Cartão de Crédito

Trabalho de Sistemas Inteligentes. Comparação entre Regressão Logística e Random Forest
na detecção de transações fraudulentas, avaliados por *holdout* 80/20 e validação cruzada
*k-fold* (k = 5).

## Base de dados

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) —
transações de cartões europeus em setembro de 2013.

- 284.807 transações × 31 colunas (283.726 após remover duplicatas)
- 473 fraudes — **0,17%** das transações, ou 1 a cada 600
- `V1`–`V28` são componentes de PCA aplicados pelos autores por confidencialidade;
  apenas `Time`, `Amount` e `Class` estão na forma original

O CSV tem 143 MB. Baixe pelo link 
acima e coloque em `dados/creditcard.csv`.

## Tratamento

- Remoção de 1.081 linhas duplicadas
- Remoção da coluna `Time` — são segundos desde a primeira transação da base, que não generaliza para dados futuros
- Padronização com `StandardScaler`, ajustado apenas no treino (e dentro do `Pipeline`,
  na validação cruzada, para não vazar dados)
- Desbalanceamento tratado com `class_weight="balanced"`, sem reamostragem

Restam 29 variáveis preditoras.

## Resultados

Métrica principal: **PR-AUC**. Acurácia não serve — prever "legítima" para tudo já acerta 99,83%.

### Holdout 80/20

| Modelo | AUC-PR | AUC-ROC | Recall | Precision | F1 |
|---|---|---|---|---|---|
| Regressão Logística | 0,6752 | 0,9648 | 0,8737 | 0,0558 | 0,1049 |
| Random Forest | **0,8149** | 0,9421 | 0,7263 | 0,9324 | 0,8166 |

### K-fold (k = 5), média ± desvio padrão

| Modelo | AUC-PR | AUC-ROC | Recall | Precision | F1 |
|---|---|---|---|---|---|
| Regressão Logística | 0,7540 ± 0,0271 | 0,9817 ± 0,0084 | 0,9073 ± 0,0291 | 0,0582 ± 0,0059 | 0,1093 ± 0,0102 |
| Random Forest | **0,8393 ± 0,0273** | 0,9671 ± 0,0130 | 0,8095 ± 0,0449 | 0,8980 ± 0,0211 | 0,8507 ± 0,0268 |

### Custo prático no conjunto de teste (95 fraudes)

| Modelo | Fraudes que escaparam | Clientes legítimos bloqueados |
|---|---|---|
| Regressão Logística | 12 | 1.404 |
| Random Forest | 26 | 5 |

## Conclusões

**As duas métricas discordam sobre o vencedor.** O ROC-AUC aponta a Regressão Logística
(0,9648 contra 0,9421); o AUC-PR aponta o Random Forest (0,8149 contra 0,6752). A explicação
está na classe rara: o ROC-AUC é dominado pelos ~56 mil negativos do conjunto de teste, onde
os dois modelos acertam trivialmente, e por isso não enxerga a diferença que importa. O AUC-PR
olha só para a classe positiva. É o motivo de ele ser a métrica de decisão aqui.

**A vantagem do Random Forest não é sorte da divisão.** As cinco dobras do k-fold do RF
(0,785 a 0,866) não têm nenhuma sobreposição com as cinco da Regressão Logística
(0,710 a 0,791) cinco de cinco divisões, o mesmo vencedor.

**O holdout foi pessimista com a Regressão Logística.** Seu AUC-PR de 0,6752 numa divisão
única fica abaixo de *todas* as suas cinco dobras (mínimo 0,710). Aquela divisão específica
calhou de ser desfavorável, o que ilustra por que uma estimativa isolada é frágil e por que
vale reportar as duas avaliações.

**O gargalo da Regressão Logística é precisão, não recall.** Ela encontra mais fraudes que o
RF (0,874 contra 0,726), mas com precisão de 5,6%: 1.404 alarmes falsos para 83 fraudes
encontradas. Na prática o Random Forest deixa passar 14 fraudes a mais e, em troca, evita
1.399 bloqueios indevidos — cerca de 100 clientes poupados por fraude adicional perdida.

**Modelo escolhido: Random Forest.**

## Como executar

```bash
pip install pandas numpy scikit-learn matplotlib
jupyter notebook analise_fraude.ipynb
```

## Arquivos

- `analise_fraude.ipynb` — notebook completo: carga, tratamento, holdout, k-fold e gráficos
