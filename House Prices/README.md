# HOUSE PRICES - ACCURACY ~ 0.13487
Projeto não terminado, critério para o término seria quando o **ACCURACY ~ 0.123**.

## Resumo
Nesse repositório, Treinei a famosa base do **House Prices**, usando Machine Learning.

Quando estou codificando, gosto de dividir em partes, que são: Importação de Bibliotecas básicas (onde uso em elas em diversos pontos do código), Visualização das Variáveis/Resumo Estatístico, Pré-processamento parte 1 (momento que eu limpo, arrumo, junto com critérios as colunas), Estatística, Pré-processamento parte 2 (quando testo alguns algoritmos para modificar as colunas e para analisar seus comportamentos em relação ao modelo), Modelo e finalmente Kaggle (onde eu faço as modificações na base teste e testo o modelo).

## Algoritmos em cada parte do código.
Tem pré-processamento, que não foi usado algoritmo, porém, foi usado lógica de programação e também bibliotecas.


### Importando base.
- Shuffle (Ele bagunça a ordem da linha)

### Estatística
- Correlação linear
- Predictive Power Score PPS

### Pré-processamento parte 2
- LOGe(x+1)
- Detecção de Outliers (Auto Encoder, IsolationForest)
- Escolhendo colunas (XGBRegressor)
- PCA (redução de dimensão)
- Normalização

### Modelo
- LazyRegressor (Auto-ML)
- GradientBoostingRegressor
- GridSearchCV (Como Turning)
- Export_graphviz (Para Observar a TREE)
