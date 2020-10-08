#Importando Bibliotecas Bases
import pandas as pd
import numpy as np

#Importando Dados
base= pd.read_csv("train.csv")           
base.head()

# Visualização das Variaveis
# Resumo estatistica
base.describe()

# Contando Registros
print(base["Pclass"].value_counts())
print("\n")
print(base["Sex"].value_counts())
print("\n")
print(base["Age"].value_counts())
print("\n")
print(base["SibSp"].value_counts())
print("\n")
print(base["Parch"].value_counts())
print("\n")
print(base["Cabin"].value_counts())

# Observando Valores Nulos
print(base.isnull().sum())

# Sabendo o Numero de registros da classe
base["Survived"].value_counts()


#Pre Processamento
#Excluindo colonas
ase= base.drop(labels="Cabin", axis= 1)                  
base= base.drop(labels="Ticket", axis= 1)                
base= base.drop(labels="Fare", axis= 1)                 
base= base.drop(labels="Embarked", axis= 1)             
base.head()

#Definindo se tem Familia
base["Family"]= "NaN" 

# Somando as colunas
for i in range(len(base)):
  base["Family"][i]= base["SibSp"][i]+base["Parch"][i]

# Separando com ou sem Familia
base["Family"].loc[base.Family == 0] = 0
base["Family"].loc[base.Family > 0] = 1

# Eliminando coluna
base= base.drop(labels= "SibSp", axis= 1)
base= base.drop(labels= "Parch", axis= 1)

print(base.head())
print(base["Family"].value_counts())


# Trabalhando o Nome
# Separando o Nome
nome= base["Name"]

# Pegando os pronomes de tratamentos
lista_pronomes= []
for i in range(len(nome)):
   lista_nomes= nome[i].split(", ")  
   lista_nomes= lista_nomes[1]
   lista_nomes= lista_nomes.split(". ")
   lista_nomes= lista_nomes[0]
   lista_pronomes.append(lista_nomes)

lista_pronomes_final= set(lista_pronomes)

for i in lista_pronomes_final:
  print("Numero de vezes que repetiu o %s, %i vezes" %(i, lista_pronomes.count(i)))

for i in range(len(base["Name"])):

  pronome= base["Name"][i].split(", ")  
  pronome= pronome[1]
  pronome= pronome.split(". ")
  pronome= pronome[0]

  if pronome == "Mrs" or pronome == "Miss":
    base["Name"][i] = 0

  elif pronome == "Mr" or pronome == "Master":
    base["Name"][i] = 1

  else:
    base["Name"][i] = 2

base["Name"].value_counts()

#Tratando os NAN e eliminando os outros NAN
idade_media= base["Age"].median()   
base.update(base["Age"].fillna(idade_media))

base= base.dropna()
base.head()

#Colocando variaveis em norminal em numerico
base.loc[base.Sex == "male", 'Sex'] = 0
base.loc[base.Sex == "female", 'Sex'] = 1
base.head()

#Separando as variaveis previsores e classes
previsores = base.iloc[:,[2,3,4,5,6]]
classes= base.iloc[:,1:2]
previsores.head()

#Tratando Dados Idade
# Histograma
import matplotlib.pyplot as plt
from math import log10
idade= previsores["Age"]
bins= int(round(1+3.322*log10(idade.shape[0]),1))     # formula de sturges

plt.hist(idade, bins= bins)
plt.title("Histrograma")
plt.xlabel("Idade")
plt.ylabel("Frequencia")
plt.show()

valores_bins= np.histogram(idade, bins=bins)[1]
for i in range(len(valores_bins)):
  valores_bins[i]= round(valores_bins[i],2)

print("Valores dos Bins:", "\n")
print(valores_bins)

for i in range(previsores.shape[0]):
  if previsores["Age"][i] <= valores_bins[1] :
    previsores["Age"][i] = 0
  elif previsores["Age"][i] > valores_bins[1] and previsores["Age"][i] <= valores_bins[2]:
    previsores["Age"][i] = 1
  elif previsores["Age"][i] > valores_bins[2] and previsores["Age"][i] <= valores_bins[3]:
    previsores["Age"][i] = 2
  elif previsores["Age"][i] > valores_bins[3] and previsores["Age"][i] <= valores_bins[4]:
    previsores["Age"][i] = 3
  elif previsores["Age"][i] > valores_bins[4] and previsores["Age"][i] <= valores_bins[5]:
    previsores["Age"][i] = 4
  elif previsores["Age"][i] > valores_bins[5] and previsores["Age"][i] <= valores_bins[6]:
    previsores["Age"][i] = 5
  elif previsores["Age"][i] > valores_bins[6] and previsores["Age"][i] <= valores_bins[7]:
    previsores["Age"][i] = 6
  elif previsores["Age"][i] > valores_bins[7] and previsores["Age"][i] <= valores_bins[8]:
    previsores["Age"][i] = 7
  else:
    previsores["Age"][i] = 8

previsores["Age"]=previsores["Age"].astype('int64')
print(previsores["Age"].value_counts())

#Equilibrando os dados
from imblearn.over_sampling import SMOTE

shape_anterior = classes["Survived"].shape[0]

oversample = SMOTE(sampling_strategy='minority')
previsores, classes= oversample.fit_resample(previsores, classes)

#Transformando em DataFrame
previsores, classes = pd.DataFrame(previsores), pd.DataFrame(classes)

# Mudando os nomes das colonas
previsores.rename(columns={0:"Pclass", 1: "Name",2:"Sex", 3:"Age", 4:'Family'}, inplace = True)
classes.rename(columns={0:"Survived"}, inplace = True)

print("Valores da classes", "\n", classes["Survived"].value_counts(),"\n")
print("Novos registros: %i" %(classes["Survived"].shape[0]-shape_anterior))


#Diminuição de dimensão
teste_dimensao= np.copy(previsores)
teste_dimensao= np.array(teste_dimensao)
teste_dimensao

from sklearn.decomposition import PCA
pca = PCA(n_components = None)
pca.fit(teste_dimensao)

print((pca.explained_variance_ratio_))
print("Usando apens duas dimensões temos {}%".format(pca.explained_variance_ratio_[0]+pca.explained_variance_ratio_[1]))

# Caso queira diminuir as dimensoes
pca_oficial= PCA(n_components= 2)
previsores= pd.DataFrame(pca_oficial.fit_transform(np.array(previsores)))
previsores.head()

#Polynomial Features
from sklearn.preprocessing import PolynomialFeatures 
poly= PolynomialFeatures(degree= 2)
previsores= poly.fit_transform(previsores)

previsores= pd.DataFrame(previsores)
previsores.head()

#Normalização dos Dados
from sklearn.preprocessing import Normalizer
normalizador= Normalizer()
previsores= normalizador.fit_transform(previsores)
previsores= pd.DataFrame(previsores)
previsores.head()

#One-Hot encoding
pd.DataFrame(previsores).head()

a= np.copy(previsores)
print(a)

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

oneHot= ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [0,1,2,4])],remainder='passthrough')
a= oneHot.fit_transform(a)
oneHot= ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [10])],remainder='passthrough')
a= oneHot.fit_transform(a)


print(a.shape)

#Model
#Modelo de Algoritmos

# Baixar a biblioteca
!pip install lazypredict

from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test= train_test_split(previsores, classes, test_size= 0.1, random_state= 1)

lazy= LazyClassifier()

modelos, predicoes= lazy.fit(x_train, x_test, y_train, y_test)
modelos

# Top 5 Modelos
modelos= predicoes.index
for i in range(5):
  print(modelos[i])

# Preparando os dados
previsores= np.array(previsores)
classes= np.array(classes)

## Modelo LGBMClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV

# Modelo LGBMClassifier
classificador_LGBMClassifier= LGBMClassifier(objective= "binary")

# Criando parametros
parametros_LGBMClassifier= {"boosting_type": ["gbdt", "dart", "goss", "rf"],                      # 4
                            "learning_rate": [round(i,2) for i in np.arange(0, 0.21, 0.05)],      # 5
                            "n_estimators": [i for i in range(25,150, 25)]}                       # 5

# criando
greidSearch_LGBMClassifier= GridSearchCV(estimator= classificador_LGBMClassifier,
                          param_grid= parametros_LGBMClassifier,         
                          scoring= "accuracy",                         
                          cv=10)                                       
                          
# Treinamento
greidSearch_LGBMClassifier=greidSearch_LGBMClassifier.fit(previsores, classes)

# Resultados
melhores_paramentros= greidSearch_LGBMClassifier.best_params_
melhor_precisao= greidSearch_LGBMClassifier.best_score_

print("Melhores parametros foram:", melhores_paramentros)
print("E sua Precisao foi de:", melhor_precisao)

## Modelo xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

# Modelo
classificador_xgboost= XGBClassifier()

# Criando parametros
parametros_xgboost= {"max_depth": [i for i in range(2, 12, 2)],                
                     "n_estimators": [x for x in range(10, 60, 10)],           
                     "learning_rate": [j for j in np.arange(0.1, 1.1, 0.1)]    
                      }

# criando
greidSearch_xgboost= GridSearchCV(estimator= classificador_xgboost,        
                                  param_grid= parametros_xgboost,          
                                  scoring= "accuracy",                     
                                  cv=10)                                                            
# Treinamento
greidSearch_xgboost=greidSearch_xgboost.fit(previsores, classes)

# Resultados
melhores_paramentros= greidSearch_xgboost.best_params_
melhor_precisao= greidSearch_xgboost.best_score_

print("Melhores parametros foram:", melhores_paramentros)
print("E sua Precisao foi de:", melhor_precisao)


## Modelo KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# Modelo KNeighborsClassifier
classificador_KNeighborsClassifier= KNeighborsClassifier()

# Criando parametros
parametros_KNeighborsClassifier= {"n_neighbors": [i for i in range(1, 11)],              
                                  "weights": ["uniform", "distance"],                    
                                  "algorithm": ["auto", "ball_tree", "kd_tree", "brute"]  
                                  }

# criando
greidSearch_KNeighborsClassifier= GridSearchCV(estimator= classificador_KNeighborsClassifier,
                                  param_grid= parametros_KNeighborsClassifier,         
                                  scoring= "accuracy",                         
                                  cv=10)                                       
                          
# Treinamento
greidSearch_KNeighborsClassifier=greidSearch_KNeighborsClassifier.fit(previsores, classes)

# Resultados
melhores_paramentros= greidSearch_KNeighborsClassifier.best_params_
melhor_precisao= greidSearch_KNeighborsClassifier.best_score_

print("Melhores parametros foram:", melhores_paramentros)
print("E sua Precisao foi de:", melhor_precisao)


## Modelo Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Modelo RandomForest
classificador_randomForest= RandomForestClassifier()

# Criando parametros
parametros_randomForest= {"max_depth": [i for i in range(2,12,2)],                          
                          "n_estimators": [x for x in range(5, 55, 10)],                    
                          "criterion": ["gini", "entropy"],                                 
                          "max_features": ['auto', 'sqrt', 'log2']}                         

# criando
greidSearch_randomForest= GridSearchCV(estimator= classificador_randomForest,
                          param_grid= parametros_randomForest,         
                          scoring= "accuracy",                         
                          cv=10)                                       
                          
# Treinamento
greidSearch_randomForest=greidSearch_randomForest.fit(previsores, classes)

# Resultados
melhores_paramentros= greidSearch_randomForest.best_params_
melhor_precisao= greidSearch_randomForest.best_score_

print("Melhores parametros foram:", melhores_paramentros)
print("E sua Precisao foi de:", melhor_precisao)


## Modelo RidgeClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import GridSearchCV

# Modelo RidgeClassifier
classificador_RidgeClassifier= RidgeClassifier()

# Criando parametros
parametros_RidgeClassifier= {"solver": ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]}  # 7

# criando
greidSearch_RidgeClassifier= GridSearchCV(estimator= classificador_RidgeClassifier,
                          param_grid= parametros_RidgeClassifier,         
                          scoring= "accuracy",                         
                          cv=10)                                       
                          
# Treinamento
greidSearch_RidgeClassifier=greidSearch_RidgeClassifier.fit(previsores, classes)

# Resultados
melhores_paramentros= greidSearch_RidgeClassifier.best_params_
melhor_precisao= greidSearch_RidgeClassifier.best_score_

print("Melhores parametros foram:", melhores_paramentros)
print("E sua Precisao foi de:", melhor_precisao)


## Modelo gradientBoosting
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

# Modelo
classificador_gradientBoosting= GradientBoostingClassifier()

# Criando parametros
parametros_gradientBoosting= {"max_depth": [i for i in range(2,12, 2)],               
                              "n_estimators": [x for x in range(5, 50, 5)],           
                              "learning_rate": [j for j in np.arange(0.1, 1.1, 0.1)]  
                              }

# criando
greidSearch_gradientBoosting= GridSearchCV(estimator= classificador_gradientBoosting,  
                                  param_grid= parametros_gradientBoosting,             
                                  scoring= "accuracy",                                 
                                  cv=10)                                                                         
# Treinamento
greidSearch_gradientBoosting= greidSearch_gradientBoosting.fit(previsores, classes)

# Resultados
melhores_paramentros= greidSearch_gradientBoosting.best_params_
melhor_precisao= greidSearch_gradientBoosting.best_score_

print("Melhores parametros foram:", melhores_paramentros)
print("E sua Precisao foi de:", melhor_precisao)

#Modelo de Votação
from sklearn.ensemble import VotingClassifier

classificador = VotingClassifier(estimators=[("RandomForest", greidSearch_randomForest.best_estimator_),
                                             ("Xgboost", greidSearch_xgboost.best_estimator_),
                                             ("GradientBoosting", greidSearch_gradientBoosting.best_estimator_),
                                             ("LGBMClassifier",greidSearch_LGBMClassifier.best_estimator_),
                                             ("KNeighborsClassifier",greidSearch_KNeighborsClassifier.best_estimator_),
                                             ("RidgeClassifier",greidSearch_RidgeClassifier.best_estimator_)])
classificador.fit(previsores, classes)


print("Seu Score Total é: %f" %(classificador.score(previsores, classes)))


#Preparando o Test_kaggle
# Carregamento da base
test= pd.read_csv("test.csv")           

# Excluir colonas inuteis
test= test.drop(labels="Cabin", axis= 1)
test= test.drop(labels="Ticket", axis= 1)    
test= test.drop(labels="Fare", axis= 1)                 
test= test.drop(labels="Embarked", axis= 1)             
test.head()

#Definindo se tem Familia
test["Family"]= "NaN"        

# Somando as colunas
for i in range(len(test)):
  test["Family"][i]= test["SibSp"][i]+test["Parch"][i]

# Separando com ou sem Familia
test["Family"].loc[test.Family == 0] = 0
test["Family"].loc[test.Family > 0] = 1

# Eliminando coluna
test= test.drop(labels= "SibSp", axis= 1)
test= test.drop(labels= "Parch", axis= 1)

print(test.head())
print(test["Family"].value_counts())


#Trabalhando o Nome
# Separando o Nome
nome= test["Name"]

# Pegando os pronomes de tratamentos
lista_pronomes= []
for i in range(len(nome)):
   lista_nomes= nome[i].split(", ")
   lista_nomes= lista_nomes[1]
   lista_nomes= lista_nomes.split(". ")
   lista_nomes= lista_nomes[0]
   lista_pronomes.append(lista_nomes)

lista_pronomes_final= set(lista_pronomes)

for i in lista_pronomes_final:
  print("Numero de vezes que repetiu o %s, %i vezes" %(i, lista_pronomes.count(i)))

for i in range(len(test["Name"])):

  pronome= test["Name"][i].split(", ")  
  pronome= pronome[1]
  pronome= pronome.split(". ")
  pronome= pronome[0]

  if pronome == "Mrs" or pronome == "Miss":
    test["Name"][i] = 0

  elif pronome == "Mr" or pronome == "Master":
    test["Name"][i] = 1

  else:
    test["Name"][i] = 2

test["Name"].value_counts()

#Tratando os NAN e eliminando os outros NAN
# Colocando variaveis em norminal em numerico
test.loc[test.Sex == "male", 'Sex'] = 0
test.loc[test.Sex == "female", 'Sex'] = 1

# Tratar Valores NAN
print("ANTES")
print(test.isnull().sum())     

idade_media= test["Age"].median()  
test.update(test["Age"].fillna(idade_media))

print("\n")
print("DEPOIS")
print(test.isnull().sum())

#Tratando Dados Idade
# Histograma
import matplotlib.pyplot as plt
from math import log10
idade= test["Age"]
bins= int(round(1+3.322*log10(idade.shape[0]),1))     # formula de sturges

plt.hist(idade, bins= bins)
plt.title("Histrograma")
plt.xlabel("Idade")
plt.ylabel("Frequencia")
plt.show()

valores_bins= np.histogram(idade, bins=bins)[1]
for i in range(len(valores_bins)):
  valores_bins[i]= round(valores_bins[i],2)

print("Valores dos Bins:", "\n")
print(valores_bins)

for i in range(test.shape[0]):
  if test["Age"][i] <= valores_bins[1] :
    test["Age"][i] = 0
  elif test["Age"][i] > valores_bins[1] and test["Age"][i] <= valores_bins[2]:
    test["Age"][i] = 1
  elif test["Age"][i] > valores_bins[2] and test["Age"][i] <= valores_bins[3]:
    test["Age"][i] = 2
  elif test["Age"][i] > valores_bins[3] and test["Age"][i] <= valores_bins[4]:
    test["Age"][i] = 3
  elif test["Age"][i] > valores_bins[4] and test["Age"][i] <= valores_bins[5]:
    test["Age"][i] = 4
  elif test["Age"][i] > valores_bins[5] and test["Age"][i] <= valores_bins[6]:
    test["Age"][i] = 5
  elif test["Age"][i] > valores_bins[6] and test["Age"][i] <= valores_bins[7]:
    test["Age"][i] = 6
  elif test["Age"][i] > valores_bins[7] and test["Age"][i] <= valores_bins[8]:
    test["Age"][i] = 7
  elif test["Age"][i] > valores_bins[8] and test["Age"][i] <= valores_bins[9]:
    test["Age"][i] = 8
  else:
    test["Age"][i] = 9
  
print(test["Age"].value_counts())

# Passageiros
passageiro= test.iloc[:,0:1]

# Drop
test= test.drop("PassengerId", axis= 1)


#PreProcesso
# Caso queira diminuir as dimensoes
test= pd.DataFrame(pca_oficial.fit_transform(np.array(test)))
test.head()

# Polinomial
test= poly.fit_transform(test)

test= pd.DataFrame(test)
test.head()

# One-Hot
test= oneHot.transform(np.array(test))
test= pd.DataFrame(test)
test.head()

# Predict
teste_final= classificador.predict(np.array(test))

# Transformando em dataframe e mudando o nome
teste_final = pd.DataFrame(teste_final)
teste_final.columns= ["Survived"]
teste_final.head()

# Colocando duas DataFrame em Uma
x= pd.concat([passageiro, teste_final], axis= 1)
x.head()

# index
x.set_index('PassengerId', inplace=True)
x

# Convertendo em CSV
x.to_csv("Voting1.csv", sep=',', encoding='utf-8')
