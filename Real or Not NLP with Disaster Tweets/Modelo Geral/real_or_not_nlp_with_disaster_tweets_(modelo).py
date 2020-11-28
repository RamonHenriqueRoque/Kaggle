# -*- coding: utf-8 -*-
"""
#**Importar bibliotecas basicas**
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
plt.rcParams['figure.figsize'] = (15,12)                # arrumar o tamanho

import random
random.seed(8)

import warnings
warnings.filterwarnings('ignore')                 # Eliminar mensagens

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

"""#**Importando bases**"""

# Pegando do google Drive
from google.colab import drive
import zipfile                                                                               # abrir pastas compactadas 

drive.mount("/content/gdrive")                                                               # DEVINIR AONDE VC QUER SALVAR

caminho = "/content/gdrive/My Drive/PYTHON/Base Preprocessamento de Real or Not Tweets.zip"  # mostrar o caminho do arquivo
zip_objeto= zipfile.ZipFile(caminho, mode='r')                                               # caminho do arquivo, mode= "r" => read
zip_objeto.extractall('./')                                                                  # onde eu quero que tenha os materias
zip_objeto.close()                                                                           # para liberar memorias

# Importando base
base_Vetor_palavras_lemmatization= pd.read_csv("Base Preprocessamento de Real or Not Tweets (VETOR PALAVRAS - Lemma).csv", dtype= "uint8")
base_Vetor_palavras_stemming= pd.read_csv("Base Preprocessamento de Real or Not Tweets (VETOR PALAVRAS - Stemming).csv", dtype= "uint8")

base_TFIDF_lemmatization= pd.read_csv("Base Preprocessamento de Real or Not Tweets (TFIDF - Lemma).csv", dtype= "uint8")
base_TFIDF_stemming= pd.read_csv("Base Preprocessamento de Real or Not Tweets (TFIDF - Stemming).csv", dtype= "uint8")

classes= pd.read_csv("Classes de Real or Not Tweets.csv", dtype= "uint8")

# Apagando Coluna
base_TFIDF_lemmatization.drop(columns= ["Unnamed: 0"], inplace= True)
base_TFIDF_stemming.drop(columns= ["Unnamed: 0"], inplace= True)
base_Vetor_palavras_lemmatization.drop(columns= ["Unnamed: 0"], inplace= True)
base_Vetor_palavras_stemming.drop(columns= ["Unnamed: 0"], inplace= True)
classes.drop(columns= ["Unnamed: 0"], inplace= True)

"""# **Modelos**

## **Rede Neural Artificial**
"""

# Arrumando a classe
from sklearn.preprocessing import OneHotEncoder

oneHot_classes= OneHotEncoder().fit(np.array(classes).reshape(-1,1))
classes_RNA= oneHot_classes.transform(np.array(classes).reshape(-1,1)).toarray()
classes_RNA= pd.DataFrame(classes_RNA)
classes_RNA.rename(columns={0:"Falso", 1: "Verdadeiro"}, inplace = True)
classes_RNA.head()

from keras.callbacks import ModelCheckpoint
from keras.models import Sequential  
from keras.layers import Dense, Dropout

previsores= base_Vetor_palavras_lemmatization

def redeneural():

    classificador= Sequential()
    
    classificador.add(Dense(units= 64,                       
                            activation="relu",             
                            kernel_initializer= "random_uniform",
                            input_dim= previsores.shape[1]))                       

    classificador.add(Dropout(0.25))

    classificador.add(Dense(units= 32,                       
                            activation="relu",             
                            kernel_initializer= "random_uniform"))

    classificador.add(Dropout(0.25))
    
    ## SAIDA
    classificador.add(Dense(units= classes_RNA.shape[1],    
                            activation="sigmoid"))    
    ## compile
    classificador.compile(optimizer= "adam",          
                          loss= "binary_crossentropy",
                          metrics= ["accuracy"])       
    return classificador


# Callback
## SALVAR O MODELO EM CADA UMA DAS EPOCAS
mcp_accuracy= ModelCheckpoint(filepath= "pesos_accuracy_Artificial.h5",
                              monitor= "accuracy", 
                              save_best_only= True,  
                              verbose= 1)           

#TREINAMENTO
from keras.wrappers.scikit_learn import KerasClassifier
classificador= KerasClassifier(build_fn= redeneural,
                               epochs= 10,           
                               batch_size= 2)
                               
# Validação Cruzada 
from sklearn.model_selection import cross_val_score
resultados= cross_val_score(estimator= classificador,
                            X=np.array(previsores),            
                            y=np.array(classes_RNA),               
                            cv=10,
                            fit_params={'callbacks': [mcp_accuracy]})

print(resultados)
print((pd.DataFrame(resultados)*100).describe())

# Salva Arquitetura Rede Neural
classificador_json= redeneural().to_json()  
with open("RedeNeuralArtificial_Tweets.json", "w") as json_file:
    json_file.write(classificador_json)

# Carregando Rede Natural
from keras.models import model_from_json

arquivo= open("RedeNeuralArtificial_Tweets.json", "r")
estrutura_rede= arquivo.read()
arquivo.close()

classificador= model_from_json(estrutura_rede)
classificador.load_weights('pesos_accuracy_Artificial.h5')

# Testando o modelo com a base que foi treinada
## Predict
precisao= pd.DataFrame(classificador.predict(np.array(previsores)))

## Mudando Nome
precisao.rename(columns={0:"Falso", 1: "Verdadeiro"}, inplace = True)

## Fazendo criterio
precisao.loc[precisao.Falso >= 0.5, 'Falso'] = 1.0
precisao.loc[precisao.Falso < 0.5, 'Falso'] = 0.0

precisao.loc[precisao.Verdadeiro >= 0.5, 'Verdadeiro'] = 1.0
precisao.loc[precisao.Verdadeiro < 0.5, 'Verdadeiro'] = 0.0

## Prob de acerto do alg.
from sklearn.metrics import confusion_matrix, accuracy_score
matriz_confusao= confusion_matrix(oneHot_classes.inverse_transform(classes_RNA), oneHot_classes.inverse_transform(np.array(precisao)))
score = accuracy_score(classes_RNA, np.array(precisao))

print("Precisão =", round(score*100,1), "%\n")

sns.heatmap(matriz_confusao, annot= True, cbar= False, fmt="d")
plt.title("MATRIZ DE CONFUSÃO", fontsize= 15)
plt.show()

"""##**Rede Neural Convolucional**"""

# Arrumando a classe
from sklearn.preprocessing import OneHotEncoder

oneHot_classes= OneHotEncoder().fit(np.array(classes).reshape(-1,1))
classes_RNC= oneHot_classes.transform(np.array(classes).reshape(-1,1)).toarray()
classes_RNC= pd.DataFrame(classes_RNC)
classes_RNC.rename(columns={0:"Falso", 1: "Verdadeiro"}, inplace = True)
classes_RNC.head()

from keras.callbacks import ModelCheckpoint
from keras.models import Sequential  
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv1D, MaxPool1D
from keras.layers import Input, Reshape
from keras.layers.normalization import BatchNormalization

previsores= base_Vetor_palavras_lemmatization
previsores= np.array(previsores).reshape(previsores.shape[0], previsores.shape[1], 1)           # Arrumando o shape

def redeneural():

    classificador= Sequential()

    ## Operador de Convolução - Etapa 1
    classificador.add(Conv1D(32,                                         # Detectores de caracteristica (indicado colocar 64, SEMPRE 2^x)
                             3,                                          # Tamanho do kernel 3
                             activation= "relu",                         # Função de ativação
                             input_shape=(previsores.shape[1], 1)
                             ))      


    ## Melhoria da Etapa 1 (colocar a escala entre 0 a 1)
    classificador.add(BatchNormalization())

    ## Pooling - Etapa 2
    classificador.add(MaxPool1D(pool_size= 2))      # Tamanho da janela que vai selecionar os maiores valores

    ## CRIAR OUTRA CAMADA DE CONVOLUÇÃO
    classificador.add(Conv1D(32, 3, activation= "relu"))
    classificador.add(BatchNormalization())
    classificador.add(MaxPool1D(pool_size= 2)) 


    ## Flatten - Etapa 3
    classificador.add(Flatten())

    ## Rede neural Densa - Etapa 4
    ### PRIMEIRA CAMADA OCULTA E A ENTRADA
    classificador.add(Dense(units=128,                              # NUMEROS DE NEURONIOS NA CAMADA OCULTAS
                            activation="relu",                      # FUNÇÃO DE ATIVAÇÃO
                            kernel_initializer= "random_uniform"))  # OS PESOS VÂO SER GERADOS ALEATORIOS
    ### DROPOUT
    classificador.add(Dropout(0.25))

    ### Segunda Camada Oculta e Dropout
    classificador.add(Dense(units=64,                              # NUMEROS DE NEURONIOS NA CAMADA OCULTAS
                            activation="relu",                      # FUNÇÃO DE ATIVAÇÃO
                            kernel_initializer= "random_uniform"))
    classificador.add(Dropout(0.25))


    ## SAIDA
    classificador.add(Dense(units= classes_RNC.shape[1],    
                            activation="sigmoid"))    
    ## compile
    classificador.compile(optimizer= "adam",          
                          loss= "binary_crossentropy",
                          metrics= ["accuracy"])     
       
    return classificador


# Callback
## SALVAR O MODELO EM CADA UMA DAS EPOCAS
mcp_accuracy= ModelCheckpoint(filepath= "pesos_accuracy_Convolucional.h5",
                              monitor= "accuracy", 
                              save_best_only= True,  
                              verbose= 1)           

#TREINAMENTO
from keras.wrappers.scikit_learn import KerasClassifier
classificador= KerasClassifier(build_fn= redeneural,
                               epochs= 5,           
                               batch_size= 2)
                               
# Validação Cruzada 
from sklearn.model_selection import cross_val_score
resultados= cross_val_score(estimator= classificador,
                            X= previsores,            
                            y= classes_RNC,               
                            cv=10,
                            fit_params={'callbacks': [mcp_accuracy]})

print(resultados)
print((pd.DataFrame(resultados)*100).describe())

# Salva Arquitetura Rede Neural
classificador_json= redeneural().to_json()  
with open("RedeNeuralConvolucional_Tweets.json", "w") as json_file:
    json_file.write(classificador_json)

# Carregando Rede Natural
from keras.models import model_from_json

arquivo= open("RedeNeuralConvolucional_Tweets.json", "r")
estrutura_rede= arquivo.read()
arquivo.close()

classificador= model_from_json(estrutura_rede)
classificador.load_weights('pesos_accuracy_Convolucional.h5')

# Testando o modelo com a base que foi treinada
## Predict
precisao= pd.DataFrame(classificador.predict(np.array(previsores)))

## Mudando Nome
precisao.rename(columns={0:"Falso", 1: "Verdadeiro"}, inplace = True)

## Fazendo criterio
precisao.loc[precisao.Falso >= 0.5, 'Falso'] = 1.0
precisao.loc[precisao.Falso < 0.5, 'Falso'] = 0.0

precisao.loc[precisao.Verdadeiro >= 0.5, 'Verdadeiro'] = 1.0
precisao.loc[precisao.Verdadeiro < 0.5, 'Verdadeiro'] = 0.0

## Prob de acerto do alg.
from sklearn.metrics import confusion_matrix, accuracy_score
matriz_confusao= confusion_matrix(oneHot_classes.inverse_transform(classes_RNA), oneHot_classes.inverse_transform(np.array(precisao)))
score = accuracy_score(classes_RNA, np.array(precisao))

print("Precisão =", round(score*100,1), "%\n")

sns.heatmap(matriz_confusao, annot= True, cbar= False, fmt="d")
plt.title("MATRIZ DE CONFUSÃO", fontsize= 15)
plt.show()

"""##**Auto-ML**

###**H2O**
* https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html

**Base**
"""

import h2o
from h2o.automl import H2OAutoML

h2o.init()          # Conectar o h20

classes_h2o= pd.DataFrame.copy(classes)
classes_h2o.rename(columns= {"target": "RESPOSTA_FINAL"}, inplace = True)
base_H2O= h2o.H2OFrame(pd.concat([pd.DataFrame.copy(base_Vetor_palavras_lemmatization), classes_h2o], axis= 1))

base_H2O["RESPOSTA_FINAL"]= base_H2O["RESPOSTA_FINAL"].asfactor()          # Para Classificação

"""**Modelo**"""

# Modelo
auto_ml_h2o= H2OAutoML(max_models=2, seed=1)
auto_ml_h2o.train(x= [i for i in base_H2O.columns if (i != "RESPOSTA_FINAL")], y= "RESPOSTA_FINAL", training_frame= base_H2O,
                  )

# Mostrar os resultados
auto_ml_h2o.leaderboard.head()

"""**Salvar Modelo**"""

# Salvar modelo
lista_modelo = auto_ml_h2o.leaderboard
id_modelo = list(lista_modelo['model_id'].as_data_frame().iloc[:,0])

for id in id_modelo:
  modelo_h2o = h2o.get_model(id)
  h2o.save_model(model=modelo_h2o, path="./Modelo_H2O", force=True)

# Carregar Modelo
carregando_modelo_h2o = h2o.load_model("Modelo_H2O/StackedEnsemble_AllModels_AutoML_20201120_120004")

"""**Predict**"""

# Testando o modelo com a base que foi treinada
precisao= h2o.as_list(auto_ml_h2o.predict(base_H2O))
precisao.head(10)

## Prob de acerto do alg.
from sklearn.metrics import confusion_matrix, accuracy_score
matriz_confusao= confusion_matrix(precisao["predict"], classes["target"])
score = accuracy_score(precisao["predict"], classes["target"])

print("Precisão =", round(score*100,1), "%\n")

sns.heatmap(matriz_confusao, annot= True, cbar= False, fmt="d")
plt.title("MATRIZ DE CONFUSÃO", fontsize= 15)
plt.show()

"""###**PyCaret**
* https://pycaret.org/automl/

**Modelo**
"""

from pycaret.classification import *

# Preparando o modelo
classes_pyCaret= pd.DataFrame.copy(classes)
classes_pyCaret.rename(columns= {"target": "RESPOSTA_FINAL"}, inplace = True)
base_pyCaret= pd.concat([pd.DataFrame.copy(base_Vetor_palavras_lemmatization), classes_pyCaret], axis= 1)

# Modelo
modelo_pyCaret= setup(data= base_pyCaret, target= "RESPOSTA_FINAL", train_size= 0.9, silent = True)

# TOP 5 modelo
top5= compare_models(n_select= 5)

# ajuste dos 5 principais modelos
tuned_top5 = [tune_model(i) for i in top5]

# Validação cruzada
bagged_top5 = [ensemble_model(i) for i in tuned_top5]

# votação
blender = blend_models(estimator_list = top5)

# melhor modelo
best = automl(optimize = "Accuracy")

"""**Salvando o Modelo**"""

# Salvando o Modelo
save_model(best, "Modelo_PyCaret")

# Carregando o Modelo
modelo_PyCaret= load_model("Modelo_PyCaret")

"""**Testando o modelo**"""

# Predict
predict_pycaret= predict_model(modelo_PyCaret, base_Vetor_palavras_lemmatization)
predict_pycaret[["Label", "Score"]]

# Grafico
from sklearn.metrics import confusion_matrix, accuracy_score
matriz_confusao= confusion_matrix(predict_pycaret["Label"], classes)
score = accuracy_score(predict_pycaret["Label"], classes)

print("Precisão =", round(score*100,1), "%\n")

sns.heatmap(matriz_confusao, annot= True, cbar= False, fmt="d")
plt.title("MATRIZ DE CONFUSÃO", fontsize= 15)
plt.show()

"""###**AUTO-SKLEARN**
* https://automl.github.io/auto-sklearn/master/
* https://automl.github.io/auto-sklearn/master/api.html?highlight=autosklearnclassifier#autosklearn.classification.AutoSklearnClassifier

**Modelo**
"""

import autosklearn.classification

modelo_auto_sklearn= autosklearn.classification.AutoSklearnClassifier(n_jobs= -1,                     # Nº de processadores
                                                                      time_left_for_this_task= 20*60, # Tempo de execução em geral => Segundo
                                                                      per_run_time_limit= 30,         # Tempo max de execução para cada modelo => Segundo
                                                                      seed= 4
                                                                      ).fit(base_Vetor_palavras_lemmatization,
                                                                            classes)

# Retorna as estatísticas do resultado do treinamento => Possivelmente o melhor modelo
print(modelo_auto_sklearn.sprint_statistics())

# Modelos encontrados => (ensemble weight, machine learning pipeline)
print(modelo_auto_sklearn.show_models())

"""**Testando o modelo**"""

# Predict
predict_auto_sklearn= pd.DataFrame(modelo_auto_sklearn.predict(base_Vetor_palavras_lemmatization))
predict_auto_sklearn.head()

# Grafico
from sklearn.metrics import confusion_matrix, accuracy_score
matriz_confusao= confusion_matrix(predict_auto_sklearn[0], classes)
score = accuracy_score(predict_auto_sklearn[0], classes)

print("Precisão =", round(score*100,1), "%\n")

sns.heatmap(matriz_confusao, annot= True, cbar= False, fmt="d")
plt.title("MATRIZ DE CONFUSÃO", fontsize= 15)
plt.show()

"""**Salvando Modelo**"""

import joblib

# Salvando modelo
joblib.dump(modelo_auto_sklearn, "Modelo_AutoSklearn.sav")

# Carregando modelo
modelo_auto_sklearn_load= joblib.load("Modelo_AutoSklearn.sav")
print("Precisão = %.4f" %(accuracy_score(pd.DataFrame(modelo_auto_sklearn_load.predict(base_Vetor_palavras_lemmatization))[0],classes)))

"""###**TPOT**
* http://epistasislab.github.io/tpot/

**Modelo**
"""

from tpot import TPOTClassifier

modelo_TPOT= TPOTClassifier(generations=5, population_size=100,
                            cv=5, n_jobs= -1, random_state=2, verbosity=2).fit(base_Vetor_palavras_lemmatization, classes)

my_dict = list(modelo_TPOT.evaluated_individuals_.items())

model_scores = pd.DataFrame()
for model in my_dict:
    model_name = model[0]
    model_info = model[1]
    cv_score = model[1].get('internal_cv_score') 
    model_scores = model_scores.append({'model': model_name,
                                        'cv_score': cv_score,
                                        'model_info': model_info,},
                                       ignore_index=True)

model_scores = model_scores.sort_values('cv_score', ascending=False)
model_scores.head()

"""**Grafico**"""

predict_TPOT= modelo_TPOT.predict(base_Vetor_palavras_lemmatization)

# Grafico
from sklearn.metrics import confusion_matrix, accuracy_score
matriz_confusao= confusion_matrix(predict_TPOT, classes)
score = accuracy_score(predict_TPOT, classes)

print("Precisão =", round(score*100,1), "%\n")

sns.heatmap(matriz_confusao, annot= True, cbar= False, fmt="d")
plt.title("MATRIZ DE CONFUSÃO", fontsize= 15)
plt.show()

"""### **LazyPredict**"""

from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split

previsores= np.array(base_Vetor_palavras_lemmatization)
classes_ML= np.array(pd.DataFrame.copy(classes))

x_train, x_test, y_train, y_test= train_test_split(previsores, classes_ML, test_size= 0.1, random_state= 1)

lazy= LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)

modelos, predicoes= lazy.fit(x_train, x_test, y_train, y_test)
modelos
