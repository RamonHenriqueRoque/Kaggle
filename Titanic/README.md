# TITANIC - ACCURACY ~ 79
Projeto não terminado, critério para o termino seria quando o **ACCURACY > 81**.

## Resumo
Nesse repositório, treinamos dois tipos de IA (Machine Learning e o Deep Learning) na famosa base do **Titanic**.

Os dois arquivos têm suas bases iguais (pré-processamento parte 1), o que se difere entre eles são os modelos. Quando estou codificando, gosto de dividir em partes, que são: Importação de Bibliotecas básicas (onde uso em elas em diversos pontos do código), Visualização das Variáveis/Resumo Estatístico, Pré-processamento parte 1 (momento que eu limpo, arrumo, junto com critérios as colunas), Pré-processamento parte 2 (quando testo alguns algoritmos para modificar as colunas e para analisar seus comportamentos em relação ao modelo), Modelo e finalmente Kaggle (onde eu faço as modificações na base teste e testo o modelo).

O Modelo de rede neural um resultado melhor.

## Algoritmos usados entre os dois arquivos.
- PCA (redução de dimensão)
- PolynomialFeatures 
- Normalizer
- One-Hot Encoding (Dummy)

## Algoritmos usados apenas no arquivo algoritmo 
- over sampling - SMOTE 
- LazyClassifier (Auto-ML)

## Modelo arquivo algoritmo
Foram usados os seguintes algoritmos: LGBMClassifier, XGBClassifier, KNeighborsClassifier, RandomForestClassifier, RidgeClassifier, GradientBoostingClassifier e para ter um sistema de votação VotingClassifier. Todos os algoritmos mencionados foram usados um turning para pegar os melhores atributos.

## Modelo arquivo Rede Neural
A arquitetura foi de dois neurônios ocultos, Callbacks (para pegar o melhor modelo possível).
