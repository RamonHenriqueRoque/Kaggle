# -*- coding: utf-8 -*-
"""
#**Baixando Lib**
"""

# H2o
#!pip install h2o

# Pycaret
#!pip install pycaret==2.0

# Lazy-Predict
#!pip install lazypredict

# Auto-Sklearn => Dps de instalar, reiniciar o notebook, e dps instalar de novo.
#! apt-get install swig -y
#! pip install Cython numpy
#! pip install auto-sklearn

# TPOT
#! pip install tpot

"""# **Importar Bibliotecas Basicas**"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
plt.rcParams['figure.figsize'] = (15,12)                # arrumar o tamanho

import random
random.seed(8)

import warnings
warnings.filterwarnings('ignore')                 # Eliminar mensagens

import nltk
nltk.download('stopwords')                              # Stop Words
nltk.download('wordnet')
nltk.download('punkt')                                  # POS part-of-speech
nltk.download('averaged_perceptron_tagger')             # POS part-of-speech

"""# **Importando dados**"""

base_oficial= pd.read_csv("train.csv")
base= pd.DataFrame.copy(base_oficial)

"""# **Analisando os dados**

**Observando os dados**
"""

base.head()

"""**Descrição estatisticas**"""

base.describe()

"""**Informação**"""

base.info()

"""**Contando Valores**"""

for coluna in ["keyword", "location", "target"]:
  print(base[coluna].value_counts(dropna= False), "\n", "###################"*4)

"""**Observando os valores (Valores nao se repetem)**"""

for coluna in ["keyword", "location", "target"]:
  print(base[coluna].unique(),"\n", "#######################" * 4)

"""**Analise do Target**"""

sns.countplot(x= "target", data= base)
plt.title("Valores nas base de Fake")
plt.show()

"""# **Pre-Processamento**

**Pegando apenas o ID, Texto e a Previsão**
"""

base= base[["text", "target"]]
base.head()

"""##**Limpando o Texto**"""

base["clear_text"]= pd.DataFrame.copy(base["text"])
base= base[["text", "clear_text", "target"]]
base.head()

"""**Reconhecimento de Entidade Nomeada (NER)**

NER é usado para identificar entidades nomeadas como nomes de pessoas, organizações, locais, quantidades, valores monetários, porcentagens, etc.

"""

'''# https://spacy.io/api/annotation#named-entities      pegar as tags

import spacy
pln = spacy.load('en_core_web_sm')
lista_nome_NER= []
for linha in range(len(base)):
  texto_NER= pln(base["clear_text"][linha].lower())

  for entidade in texto_NER.ents:
    if entidade.label_ == "PERSON":
      #print(entidade.text, entidade.label_)
      lista_nome_NER.append(entidade.text)

lista_nome_NER= set(lista_nome_NER)
lista_nome_NER'''

# ESTA PEGANDO COISAS ALEM DE NOME

"""**Deixar todas as palavras minusculo**"""

for index in range(len(base)):
  base["clear_text"][index]= base["clear_text"][index].lower()
base["clear_text"]

"""**StopWord**"""

from spacy.lang.en.stop_words import STOP_WORDS
from nltk.corpus import stopwords

stopWord_nltk= stopwords.words("english")
stopWord_spacy = list(STOP_WORDS)                 

stopWord= set(stopWord_nltk + stopWord_spacy)
print("Numero de StopWord =",len(stopWord))

"""**Pontuações**"""

import string
pontuacao = list(string.punctuation)      # Tiro as pontuações
print("Numeros de pontuações =", len(pontuacao))

"""**Tirando o StopWord e as pontuações**"""

def remove_stopwords(text):
  for index in range(len(text)):  
    final_text = []
    for i in text[index].split():
        if i not in stopWord and i not in pontuacao:
            final_text.append(i)
    text[index]= " ".join(final_text)
  return text
base["clear_text"]= remove_stopwords(base["clear_text"])
base["clear_text"]

"""**Pegando as caracteres especiais**"""
from string import ascii_lowercase, digits

def pegando_caracteres(base, campo):
  caracteres_todos= []

  # Rodar todas as linhas para pegar todas os caracteres
  for linha in range(len(base[campo])):
    frase_analise= [list(i) for i in base[campo][linha].split()]
    
    # Juntando as listas
    for contar_palavras in frase_analise:           # Referente a lista
      for letra in contar_palavras:                 # Referente a letra
        if letra not in caracteres_todos:
          caracteres_todos.append(letra)
  
  # Tirando numeros e letras
  alfa= list(ascii_lowercase)
  num= list(digits)

  for letra in alfa:
    if letra in caracteres_todos:
      caracteres_todos.remove(letra)

  for numero in num:
    if numero in caracteres_todos:
      caracteres_todos.remove(numero)
  
  caracteres_todos.pop(51)             # Para tirar => \

  return(caracteres_todos)

caracteres_especiais= pegando_caracteres(base, "clear_text")
print("Os caracteres achado foram:\n{}".format("".join(caracteres_especiais)))

# Removendo as caracteres que a lib RE reserva
pontuacoes_especiais_lib_RE= ["?", "|", "(", ")", "+", ".", "$", "^", "+", "[", "]", "&", "*"]

for pont in pontuacoes_especiais_lib_RE:
  if pont in caracteres_especiais:
    caracteres_especiais.remove(pont)

"""**Eliminando Http, ID @ e outras coisas**"""

# Filtros
import re
for index in range(len(base)):
  frase= re.findall(r"asd", base["clear_text"][index])
  if frase != []:
    print(base["clear_text"][index], "########", frase, "\n")

# Ver melhor essa biblioteca
from string import ascii_lowercase

def eliminando_texto(texto):
  for index in range(len(texto)):
    # Trocando palavras
    texto[index]=texto[index].replace(r"&amp",r'and')                            # Substituir AMP por AND
    texto[index]=texto[index].replace(r"&lt",r"<")                               # Substituir LT por <
    texto[index]=texto[index].replace(r"&gt",r">")                               # Substituir GT por >

    #Transformar as pontuações que o lib Re são restritas
    for pont in pontuacoes_especiais_lib_RE:
      texto[index]=texto[index].replace(r"%s"%(pont),r"")                        # Todas as pontuação restritas, vão ser tiradas

    # Geral
    texto[index]= re.sub(r"@[a-z0-9$-_@.&+]+"," ", texto[index])                 # @nome_qualquer
    texto[index]= re.sub(r"https?://[A-Za-z0-9./]+"," ",texto[index])            # Http
    texto[index]= re.sub(r"<[^<]+>"," ", texto[index])                           # Removendo tags
    
    # caracter indesejaveis
    for simbolos in caracteres_especiais:
      segmento_simbolos= [r"%s[a-z0-9$-_@.&ªºòóï+]+"%(simbolos), r"%s+"%(simbolos)]
      texto[index]= re.sub(segmento_simbolos[0]," ", texto[index])
      texto[index]= re.sub(segmento_simbolos[1]," ", texto[index])                         
    
    # Repetição de letras (Regra: mais de duas letras repetidas, substitui por duas)
    abc= list(ascii_lowercase)
    for letra in abc:
      segmento_letra= r"%s"%(letra*3)
      
      if re.findall(segmento_letra, texto[index]) != []:
        while True:
          sub_segmento_letra= "%s"%(letra*2)
          texto[index]= re.sub(segmento_letra, sub_segmento_letra, texto[index])
          if re.findall(segmento_letra, texto[index]) == []:
            break

    # Repetição de duas letras no começo da frase
    for letra in abc:
      segmento_letra= [r" %s"%(letra*2), r"^%s"%(letra*2)]
      sub_segmento_letra= "%s"%(letra)
      texto[index]= re.sub(segmento_letra[0], sub_segmento_letra, texto[index])
      texto[index]= re.sub(segmento_letra[1], sub_segmento_letra, texto[index])
    
    # Risadas
    texto[index]= re.sub(r"hah[a-z]+", " haha", texto[index])                    # haha junto com alguma string
    texto[index]= re.sub(r" hah[a-z]+", " haha", texto[index])                   # haha sozinho

    for letra in abc:
      segmento_letra= r"%shaha[a-z]+"%(letra)
      sub_segmento_letra= "%s haha"%(letra)
      texto[index]= re.sub(segmento_letra, sub_segmento_letra, texto[index])     # Separando a string do haha

    # Letras sozinhas
    for letra in abc:
      segmento_letra=  [r" %s "%(letra), r"^%s "%(letra), r" %s$"%(letra)]
      texto[index]= re.sub(segmento_letra[0], " ", texto[index])
      texto[index]= re.sub(segmento_letra[1], " ", texto[index])
      texto[index]= re.sub(segmento_letra[2], " ", texto[index])
      
    # Dois espaços
    texto[index]= re.sub(r" +"," ",texto[index])                                  # Eliminando quando tiver dois espaços

  return (texto)
base["clear_text"]= eliminando_texto(base["clear_text"])
base["clear_text"].head()

"""**Eliminando Registros duplicados**"""

base.drop_duplicates(subset=["clear_text", "target"], inplace= True)
base.index= [i for i in range(len(base))]                                        # Arrumando o Index
base.head()

base.shape

"""## **Stemming**"""

base_stemming= pd.DataFrame.copy(base)
base_stemming.head(3)

from nltk.stem import PorterStemmer

def stemming(text):
  stemming= PorterStemmer()
  
  for index in range(len(text)):  
    final_text = []
    #Arrumando o texto
    for i in text[index].split():
        final_text.append(stemming.stem(i))

    text[index]= " ".join(final_text)
  return text

base_stemming["clear_text"]= stemming(base_stemming["clear_text"])
base_stemming.head()

"""## **Lemmatization**"""

base_lemmatization= pd.DataFrame.copy(base)
base_lemmatization.head()

from nltk.stem.wordnet import WordNetLemmatizer

def lematizacao(text):
  lemmatization= WordNetLemmatizer()
  
  for index in range(len(text)):  
    final_text = []
    #Arrumando o texto
    for i in text[index].split():
        final_text.append(lemmatization.lemmatize(i))

    text[index]= " ".join(final_text)
  return text

base_lemmatization["clear_text"]= lematizacao(base_lemmatization["clear_text"])
base_lemmatization.head()

"""#**Visualização dos Dados: Pré Processados**

##**CloudWord**
"""

from wordcloud import WordCloud
wc_verdadeiro= base.loc[base["target"] == 1]

wc = WordCloud(max_words=2000, width= 1920, height= 1080).generate("".join(wc_verdadeiro["clear_text"]))
plt.figure(figsize=(17,13))
plt.imshow(wc, interpolation='bilinear') 
plt.title("PALAVRAS COM AS FRASES VERDADEIRA", fontsize= 20)
plt.axis("off")
plt.show()

lista_palavras_wc_verdadeiro= pd.DataFrame(data= wc.words_.keys(), columns= ["Palavras"])
lista_palavras_wc_verdadeiro["Significância"]= wc.words_.values()

for remover in range(2000):
  if remover >= 20:
    lista_palavras_wc_verdadeiro.drop(labels= remover, inplace= True)

plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['figure.figsize'] = (15,12)                

sns.barplot(y= lista_palavras_wc_verdadeiro["Palavras"], x= lista_palavras_wc_verdadeiro["Significância"])
plt.title("TOP 20 PALAVRAS COM PRETEXTO VERDADEIROS QUE MAIS APARECEM", fontsize= 20)
plt.xlabel("SIGNIFICÂNCIA", fontsize= 15)
plt.ylabel("PALAVRAS", fontsize= 20)
plt.show()

from wordcloud import WordCloud
wc_falso= base.loc[base["target"] == 0]

wc = WordCloud(max_words=2000, width= 1920, height= 1080).generate("".join(wc_falso["clear_text"]))
plt.figure(figsize=(17,13))
plt.imshow(wc, interpolation='bilinear') 
plt.title("PALAVRAS COM AS FRASES FALSAS", fontsize= 20)
plt.axis("off")
plt.show()

lista_palavras_wc_falso= pd.DataFrame(data= wc.words_.keys(), columns= ["Palavras"])
lista_palavras_wc_falso["Significância"]= wc.words_.values()

for remover in range(2000):
  if remover >= 20:
    lista_palavras_wc_falso.drop(labels= remover, inplace= True)

plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['figure.figsize'] = (15,12)              

sns.barplot(y= lista_palavras_wc_falso["Palavras"], x= lista_palavras_wc_falso["Significância"])
plt.title("TOP 20 PALAVRAS COM PRETEXTO FALSOS QUE MAIS APARECEM", fontsize= 20)
plt.xlabel("SIGNIFICÂNCIA", fontsize= 15)
plt.ylabel("PALAVRAS", fontsize= 20)
plt.show()

"""##**POS (Part of Speech Tagging)**"""

# https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html       Pegar o que as tags significa

from nltk import word_tokenize, pos_tag

lista_pos= []

for index in range(len(base)):
  tags_pos= word_tokenize(base["clear_text"][index])
  lista_pos += nltk.pos_tag(tags_pos)

lista_pos= pd.DataFrame(lista_pos, columns= ["Palavras", "Tags"])
lista_pos["Tags"]= lista_pos["Tags"].replace({"CC" :	"Conjunção coordenativa","CD" :	"Número cardinal", "DT" :	"Determinante", "EX" :	"Existencial lá", 
                                              "FW" :	"Palavra estrangeira", "IN" :	"Preposição ou conjunção subordinada", "JJ" :	"Adjetivo", 
                                              "JJR" :	"Adjetivo comparativo", "JJS" :	"Adjetivo superlativo", "LS" :	"Marcador de item de lista", 
                                              "MD" :	"Modal", "NN" :	"Substantivo singular", "NNS" :	"Substantivo plural","NNP" : "Nome próprio singular",
                                              "NNPS" :	"Substantivo próprio plural", "PDT" :	"Predeterminador", "POS" :	"Final possessivo", 
                                              "PRP" :	"Pronome pessoal", "PRP$" :	"Pronome possessivo", "RB" :	"Advérbio", "RBR" :	"Advérbio comparativo",
                                              "RBS" :	"Advérbio superlativo", "RP" :	"Partícula", "SYM" :	"Símbolo", "TO" :	"Para", "UH" :	"Interjeição", 
                                              "VB" :	"Verbo forma básica", "VBD" :	"Verbo pretérito", "VBG" :	"Verbo gerúndio ou particípio presente",
                                              "VBN" :	"Verbo particípio passado", "VBP" :	"Verbo não 3ª pessoa do singular presente", 
                                              "VBZ" :	"Verbo 3ª pessoa do singular presente", "WDT" :	"Wh-determiner", "WP" :	"Pronome Wh", 
                                              "WP$" :	"Pronome wh possessivo", "WRB" :	"Wh-advérbio"})

lista_pos.head()

quant_tags_pos= pd.DataFrame(lista_pos["Tags"].value_counts())
quant_tags_pos.head(-1)

quant_tags_pos= pd.DataFrame(lista_pos["Tags"].value_counts())

plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['figure.figsize'] = (12,8)                

sns.barplot(y= quant_tags_pos.index, x= quant_tags_pos["Tags"])

plt.title("Tipo da palavras e sua quantidades", fontsize= 20)
plt.xlabel("QUANTIDADES", fontsize= 15)
plt.ylabel("TIPO", fontsize= 20)
plt.show()

"""**Fazer um DataFrame fazer pegar a BAG of WORD**"""

base_pos=[]
for index in range(len(base)):
  pegando_tags= word_tokenize(base["clear_text"][index])

  lista_tags_vazia=[]
  for palavras, tags in pos_tag(pegando_tags):
    lista_tags_vazia.append(tags)
  base_pos.append(" ".join(lista_tags_vazia))

base_pos= pd.DataFrame(base_pos, columns= ["Tags"])
base_pos.head()

"""# **Pre-processamento parte 2**

##**Bags of words**
"""

from sklearn.feature_extraction.text import CountVectorizer

# Modelo
vetor_palavras_lemmatization= CountVectorizer().fit(base_lemmatization["clear_text"])                 # TOKEN
vetor_palavras_stemming= CountVectorizer().fit(base_stemming["clear_text"])                           # TOKEN
vetor_palavras_pos= CountVectorizer().fit(base_pos["Tags"])                           # TOKEN

# Base
bag_word_Vetor_palavras_lemmatization= pd.DataFrame(vetor_palavras_lemmatization.transform(base_lemmatization["clear_text"]).toarray(), columns= vetor_palavras_lemmatization.get_feature_names())
bag_word_Vetor_palavras_stemming= pd.DataFrame(vetor_palavras_stemming.transform(base_stemming["clear_text"]).toarray(), columns= vetor_palavras_stemming.get_feature_names())
bag_word_Vetor_palavras_pos= pd.DataFrame(vetor_palavras_pos.transform(base_pos["Tags"]).toarray(), columns= vetor_palavras_pos.get_feature_names())

# TF IDF
from sklearn.feature_extraction.text import TfidfVectorizer

# Modelo
tfidf_lemmatization = TfidfVectorizer().fit(base_lemmatization["clear_text"])
tfidf_stemming = TfidfVectorizer().fit(base_stemming["clear_text"])
tfidf_pos = TfidfVectorizer().fit(base_pos["Tags"])

# Base
bag_word_tfidf_lemmatization = pd.DataFrame(tfidf_lemmatization.transform(base_lemmatization["clear_text"]).toarray(), columns= tfidf_lemmatization.get_feature_names())
bag_word_tfidf_stemming= pd.DataFrame(tfidf_stemming.transform(base_stemming["clear_text"]).toarray(), columns= tfidf_stemming.get_feature_names())
bag_word_tfidf_pos= pd.DataFrame(tfidf_pos.transform(base_pos["Tags"]).toarray(), columns= tfidf_pos.get_feature_names())

"""##**Concatenação de bases**"""

#base_modelo= pd.concat([bag_word_Vetor_palavras_lemmatization, bag_word_Vetor_palavras_stemming, bag_word_Vetor_palavras_pos], axis= 1)
#base_modelo= pd.concat([bag_word_Vetor_palavras_lemmatization, bag_word_Vetor_palavras_pos], axis= 1)
base_modelo= pd.concat([bag_word_Vetor_palavras_lemmatization, bag_word_Vetor_palavras_stemming], axis= 1)
#base_modelo= pd.concat([bag_word_Vetor_palavras_stemming, bag_word_Vetor_palavras_pos], axis= 1)

#base_modelo= pd.concat([bag_word_tfidf_lemmatization, bag_word_tfidf_stemming, bag_word_tfidf_pos], axis= 1)
#base_modelo= pd.concat([bag_word_tfidf_lemmatization, bag_word_tfidf_pos], axis= 1)
#base_modelo= pd.concat([bag_word_tfidf_lemmatization, bag_word_tfidf_stemming], axis= 1)
#base_modelo= pd.concat([bag_word_tfidf_stemming, bag_word_tfidf_pos], axis= 1)

base_modelo.head()

"""##**Economizando espaço**"""

base_modelo= base_modelo.astype(np.uint8)
base_modelo.info()
print("\nCom o int64, estava dando 1.2 GB")

"""##**Baixando base processada**

Coloquei em outro notebook, pois cada teste de modelo, lotava a memoria RAM.

"""

'''
bag_word_Vetor_palavras_lemmatization= bag_word_Vetor_palavras_lemmatization.astype(np.uint8)
bag_word_Vetor_palavras_stemming= bag_word_Vetor_palavras_stemming.astype(np.uint8)
bag_word_tfidf_lemmatization= bag_word_tfidf_lemmatization.astype(np.uint8)
bag_word_tfidf_stemming= bag_word_tfidf_stemming.astype(np.uint8)

bag_word_Vetor_palavras_lemmatization.to_csv("Base Preprocessamento de Real or Not Tweets (VETOR PALAVRAS - Lemma).csv", sep=',', encoding='utf-8', index= "id")
bag_word_Vetor_palavras_stemming.to_csv("Base Preprocessamento de Real or Not Tweets (VETOR PALAVRAS - Stemming).csv", sep=',', encoding='utf-8', index= "id")

bag_word_tfidf_lemmatization.to_csv("Base Preprocessamento de Real or Not Tweets (TFIDF - Lemma).csv", sep=',', encoding='utf-8', index= "id")
bag_word_tfidf_stemming.to_csv("Base Preprocessamento de Real or Not Tweets (TFIDF - Stemming).csv", sep=',', encoding='utf-8', index= "id")
'''
base["target"].to_csv("Classes de Real or Not Tweets (TFIDF - Stemming).csv", sep=',', encoding='utf-8', index= "id")

"""# **Carregando Modelos**

##**Rede-Neural-Artificial**
"""

# Carregando Rede Natural
from keras.models import model_from_json

arquivo= open("RedeNeuralArtificial_Tweets.json", "r")
estrutura_rede= arquivo.read()
arquivo.close()

classificador_RNA= model_from_json(estrutura_rede)
classificador_RNA.load_weights('pesos_accuracy_Artificial.h5')

"""##**Rede-Neural_Convolucional**"""

# Carregando Rede Natural
from keras.models import model_from_json

arquivo= open("RedeNeuralConvolucional_Tweets.json", "r")
estrutura_rede= arquivo.read()
arquivo.close()

classificador_RNC= model_from_json(estrutura_rede)
classificador_RNC.load_weights('pesos_accuracy_Convolucional.h5')

"""##**Auto-ML**

###**H2O**
"""

import h2o

# Carregar Modelo
carregando_modelo_h2o = h2o.load_model("StackedEnsemble_AllModels_AutoML_20201128_182110")

"""###**PyCaret**"""

from pycaret.classification import *

# Carregando o Modelo
modelo_PyCaret= load_model("Modelo_PyCaret")

"""###**Auto-Sklearn**"""

import joblib

modelo_auto_sklearn_load= joblib.load("Modelo_AutoSklearn.sav")

"""# **Kaggle**

##**Carregando o Modelo**
"""

teste= pd.read_csv("test.csv")
id= teste["id"]
teste= teste.drop(labels= ["id", "keyword", "location"], axis= 1)
teste.head()

"""##**Limpando o Texto**"""

teste["clear_text"]= pd.DataFrame.copy(teste["text"])
teste= teste[["text", "clear_text"]]
teste.head()

"""**Deixar todas as palavras minusculo**"""

for index in range(len(teste)):
  teste["clear_text"][index]= teste["clear_text"][index].lower()
teste["clear_text"]

"""**Tirando o StopWord e as pontuações**"""

teste["clear_text"]= remove_stopwords(teste["clear_text"])
teste["clear_text"]

"""**Eliminando Http, ID @ e outras coisas**"""

teste["clear_text"]= eliminando_texto(teste["clear_text"])
teste["clear_text"]

"""## **Stemming**"""

teste_stemming= pd.DataFrame.copy(teste)
teste_stemming.head(3)

teste_stemming["clear_text"]= stemming(teste_stemming["clear_text"])
teste_stemming.head()

"""## **Lemmatization**"""

teste_lemmatization= pd.DataFrame.copy(teste)
teste_lemmatization.head()

teste_lemmatization["clear_text"]= lematizacao(teste_lemmatization["clear_text"])
teste_lemmatization

"""## **pre-processamento parte 2**

**Fazer um DataFrame fazer pegar a BAG of WORD**
"""

base_pos_teste=[]
for index in range(len(teste)):
  pegando_tags= word_tokenize(base["clear_text"][index])

  lista_tags_vazia=[]
  for palavras, tags in pos_tag(pegando_tags):
    lista_tags_vazia.append(tags)
  base_pos_teste.append(" ".join(lista_tags_vazia))

base_pos_teste= pd.DataFrame(base_pos_teste, columns= ["Tags"])
base_pos_teste.head()

"""**Bag of Word**"""

bag_word_Vetor_palavras_lemmatization_teste= pd.DataFrame(vetor_palavras_lemmatization.transform(teste_lemmatization["clear_text"]).toarray(), columns= vetor_palavras_lemmatization.get_feature_names())
bag_word_Vetor_palavras_stemming_teste= pd.DataFrame(vetor_palavras_stemming.transform(teste_stemming["clear_text"]).toarray(), columns= vetor_palavras_stemming.get_feature_names())
bag_word_Vetor_palavras_pos_teste= pd.DataFrame(vetor_palavras_pos.transform(base_pos_teste["Tags"]).toarray(), columns= vetor_palavras_pos.get_feature_names())

bag_word_tfidf_lemmatization_teste = pd.DataFrame(tfidf_lemmatization.transform(teste_lemmatization["clear_text"]).toarray(), columns= tfidf_lemmatization.get_feature_names())
bag_word_tfidf_stemming_teste = pd.DataFrame(tfidf_stemming.transform(teste_stemming["clear_text"]).toarray(), columns= tfidf_stemming.get_feature_names())
bag_word_tfidf_pos_teste = pd.DataFrame(tfidf_pos.transform(base_pos_teste["Tags"]).toarray(), columns= tfidf_pos.get_feature_names())

"""**Concatenação**"""

#base_modelo_teste= pd.concat([bag_word_Vetor_palavras_lemmatization_teste, bag_word_Vetor_palavras_stemming_teste, bag_word_Vetor_palavras_pos_teste], axis= 1)
#base_modelo_teste= pd.concat([bag_word_Vetor_palavras_lemmatization_teste, bag_word_Vetor_palavras_pos_teste], axis= 1)
base_modelo_teste= pd.concat([bag_word_Vetor_palavras_lemmatization_teste, bag_word_Vetor_palavras_stemming_teste], axis= 1)
#base_modelo_teste= pd.concat([bag_word_Vetor_palavras_stemming_teste, bag_word_Vetor_palavras_pos_teste], axis= 1)

#base_modelo_teste= pd.concat([bag_word_tfidf_lemmatization_teste, bag_word_tfidf_stemming_teste, bag_word_tfidf_pos_teste], axis= 1)
#base_modelo_teste= pd.concat([bag_word_tfidf_lemmatization_teste, bag_word_tfidf_pos_teste], axis= 1)
#base_modelo_teste= pd.concat([bag_word_tfidf_lemmatization_teste, bag_word_tfidf_stemming_teste], axis= 1)
#base_modelo_teste= pd.concat([bag_word_tfidf_stemming_teste, bag_word_tfidf_pos_teste], axis= 1)

base_modelo_teste.head()

"""## **Predict**"""

teste_predict= classificador_RNC.predict(np.array(bag_word_tfidf_lemmatization_teste))
teste_predict

"""**Retornando o valor na rede-neural normal**"""

# Transformando em dataframe e mudando o nome
teste_predict = pd.DataFrame(teste_predict)
teste_predict.columns= ["Falso", "Verdadeiro"]


# Mudando o criterio
teste_predict.loc[teste_predict.Falso >= 0.5, 'Falso'] = 1
teste_predict.loc[teste_predict.Falso < 0.5, 'Falso'] = 0

teste_predict.loc[teste_predict.Verdadeiro >= 0.5, 'Verdadeiro'] = 1
teste_predict.loc[teste_predict.Verdadeiro < 0.5, 'Verdadeiro'] = 0


teste_predict.head()

teste_predict["VALOR_REAL"] = 0

for linha in range(len(teste_predict)):
  if teste_predict["Falso"][linha] == 1:
    pass
  else:
    teste_predict["VALOR_REAL"][linha] = 1

teste_predict.head()

# Trocando o tipo da coluna
teste_predict["VALOR_REAL"]=teste_predict.VALOR_REAL.astype('int64')

# Colocando duas DataFrame em um DataFrame
x= pd.concat([id, teste_predict["VALOR_REAL"]], axis= 1)

# Index
x.set_index('id', inplace=True)

# Convertendo em CSV
x.to_csv("Real_or_not_disaster_tweets.csv", sep=',', encoding='utf-8', index= "id")
