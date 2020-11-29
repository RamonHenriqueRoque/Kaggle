**MODELO GERAL.**

# Real or Not? NLP with Disaster Tweets - ACCURACY ~ 0.7855.

## Resumo.
Dividi esse mini-projeto em duas partes (pre-processamentoo e o modelo), essa divisão de arquivo, aconteceu que tive um problema com o Google colab, então preferi fazer dois arquivos.

O primeiro arquivo é referente a analise da base, pre-processamentoo, carregando modelos e predict da base teste.

O segundo arquivo é referente ao treinamento do modelo.

### Primeiro Arquivo (Pre-Processamento)- Etapas e explicação.

- **Baixando as Libs.** - Baixando as libs que não tem baixadas no google colab.

- **Importando bibliotecas básicas e o Data Set que vai ser analisados.**

- **Analisando dados.**

=> Vejo como são os: type, descrição estatísticas, valores nulos e analisando a quantidades do target. 

- **Pré Processamentos P1.**

=> Selecionando as colunas para o analises;

=> Colocando as frase em minusculas, Tirando o StopWord e as principais pontuações, Limpeza da frase (emoticon, links, tratando as risadas, etc.);

=> Eliminando as duplicatas;

=> E por ultimo criando duas variáveis de **Stemming** e **Lemmatization**, para analisar o melhor comportamento.

- **Pre-Processamentoss P2.**

=> Bags of words - Usando os algoritmos CountVectorizer e o TfidfVectorizer, para transformar o texto em números;

=> Concatenação de bases - Tendo a ideia do Features Creation;

=> Economizando espaço - Pois os arquivos estavam ocupando muita memoria, e prejudicava o tempo de modelo e para criar os CSVs;

=> Baixando base processada - Para usar o no arquivo de MODELOS.

- **Carregando Modelos.**

=> Rede Neural Artificial;

=> Rede Neural Convolucional;

=> Auto-ML - H2O, PyCaret e Auto Skearn.

- **Kaggle.**

=> Uso o pre-processamento e o modelo na base de teste.



### Segundo Arquivo (Modelos)- Etapas e explicação.

- **Baixando as Libs.** - Baixando as libs que não tem baixadas no google colab.
- **Importando bibliotecas básicas e que foi pré processadas no primeiro arquivo**. 
- **Treinando Modelos** - Os modelos são treinados, Testados na própria base, Gravados e Carregados.

=> Rede Neural Artificial;

=> Rede Neural Convolucional;

=> Auto-ML H2O;

=> Auto-ML PyCaret;

=> Auto-ML Auto-Sklearn;

=> Auto-ML TPOT;

=> Auto-ML LazyPredict.
