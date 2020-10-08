# Importando Bibliotecas Basicas
import pandas as pd
import numpy as np

# Carregamento dos dados
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
# Excluindo colonas

base= base.drop(labels="Cabin", axis= 1)             
base= base.drop(labels="Ticket", axis= 1)            
base= base.drop(labels="Fare", axis= 1)              
base= base.drop(labels="Embarked", axis= 1)               
base.head()

# Definindo se tem Familia

base["Family"]= "NaN"        

for i in range(len(base)):
  base["Family"][i]= base["SibSp"][i]+base["Parch"][i]

base["Family"].loc[base.Family == 0] = 0
base["Family"].loc[base.Family > 0] = 1

base= base.drop(labels= "SibSp", axis= 1)
base= base.drop(labels= "Parch", axis= 1)

print(base.head())
print(base["Family"].value_counts())

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

# Tratando os NAN e eliminando os outros NAN
idade_media= base["Age"].median()    
base.update(base["Age"].fillna(idade_media))

base= base.dropna()
base.head()

# Colocando variaveis em norminal em numerico
base.loc[base.Sex == "male", 'Sex'] = 0
base.loc[base.Sex == "female", 'Sex'] = 1
base.head()

# Separando as variaveis previsores e classes
previsores = base.iloc[:,[2,3,4,5,6]]
classes= base.iloc[:,1:2]
previsores.head()

# Equilibrando os dados
from imblearn.over_sampling import SMOTE

shape_anterior = classes["Survived"].shape[0]

oversample = SMOTE(sampling_strategy='minority')
previsores, classes= oversample.fit_resample(previsores, classes)

#Transformando em DataFrame
previsores, classes = pd.DataFrame(previsores), pd.DataFrame(classes)

# Mudando os nomes das colonas
previsores.rename(columns={0:"Pclass", 1: "Name",2:"Sex", 3:"Age", 4:'SibSp'}, inplace = True)
classes.rename(columns={0:"Survived"}, inplace = True)

print("Valores da classes", "\n", classes["Survived"].value_counts(),"\n")
print("Novos registros: %i" %(classes["Survived"].shape[0]-shape_anterior))


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

print(previsores["Age"].value_counts())

# Diminuição de dimensão
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

# Polynomial Features
from sklearn.preprocessing import PolynomialFeatures 
poly= PolynomialFeatures(degree= 2)
previsores= poly.fit_transform(previsores)

previsores= pd.DataFrame(previsores)
previsores.head()

# Normalização dos Dados
from sklearn.preprocessing import Normalizer
normalizador= Normalizer()
previsores= normalizador.fit_transform(previsores)
previsores= pd.DataFrame(previsores)
previsores.head()

# One-Hot encoding
previsores.head()

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

oneHot= ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [0,1,2,3,4])],remainder='passthrough')
previsores= oneHot.fit_transform(previsores).toarray()
previsores= pd.DataFrame(previsores)
previsores.head()

oneHot_classes= ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [0])],remainder='passthrough')
classes= oneHot_classes.fit_transform(classes)
classes= pd.DataFrame(classes)
classes.head()

# Arquitetura do modelo
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.models import Sequential  
from keras.layers import Dense, Dropout
from keras import backend as k

previsores= np.array(previsores)
classes= np.array(classes)

def redeneural():
    k.clear_session()
    
    classificador.add(Dense(units=previsores.shape[1],                       
                            activation="relu",             
                            kernel_initializer= "random_uniform",
                            input_dim= 6))                       

    ## SAIDA
    classificador.add(Dense(units= classes.shape[1],    
                            activation="sigmoid"))    

    ## compile
    classificador.compile(optimizer= "adam",          
                          loss= "binary_crossentropy",
                          metrics= ["binary_accuracy"])       
    return classificador

# Callback
## SALVAR O MODELO EM CADA UMA DAS EPOCAS
mcp_accuracy= ModelCheckpoint(filepath= "pesos_accuracy.h5",
                              monitor= "binary_accuracy", 
                              save_best_only= True,  
                              verbose= 1)           

mcp_loss= ModelCheckpoint(filepath= "pesos_loss.h5",
                           monitor= "loss",         
                           save_best_only= True,    
                           verbose= 1)              

#TREINAMENTO
from keras.wrappers.scikit_learn import KerasClassifier
classificador= KerasClassifier(build_fn= redeneural,
                               epochs= 150,           
                               batch_size= 4)
                               
# Validação Cruzada 
from sklearn.model_selection import cross_val_score
resultados= cross_val_score(estimator= classificador,
                            X=previsores,            
                            y=classes,               
                            cv=10,
                            fit_params={'callbacks': [mcp_loss, mcp_accuracy]}
                            )


print(resultados)
print((pd.DataFrame(resultados)*100).describe())


# Salva Arquitetura Rede Neural
classificador_json= redeneural().to_json()  
with open("RedeNeural_Titanic.json", "w") as json_file:
    json_file.write(classificador_json)

# Carregando Rede Natural
from keras.models import model_from_json

arquivo= open("RedeNeural_Titanic.json", "r")
estrutura_rede= arquivo.read()
arquivo.close()

classificador= model_from_json(estrutura_rede)
classificador.load_weights('pesos_loss.h5')



# Testando o modelo com a base que foi treinada
## Predict
precisao= pd.DataFrame(classificador.predict(previsores))

## Mudando Nome
precisao.rename(columns={0:"Survived1", 1: "Survived2"}, inplace = True)

## Fazendo criterio
precisao.loc[precisao.Survived1 >= 0.5, 'Survived1'] = 1
precisao.loc[precisao.Survived1 < 0.5, 'Survived1'] = 0

precisao.loc[precisao.Survived2 >= 0.5, 'Survived2'] = 1
precisao.loc[precisao.Survived2 < 0.5, 'Survived2'] = 0

## Transformando em INT
precisao=precisao.astype('int64')

## Prob de acerto do alg.
from sklearn.metrics import confusion_matrix, accuracy_score
score = accuracy_score(classes, precisao)

print(round(score*100,1), "%")




# Preparando o Test_kaggle
#Carregando a Base
test= pd.read_csv("test.csv")

# Excluindo Colunas
test= test.drop(labels="Cabin", axis= 1)                
test= test.drop(labels="Ticket", axis= 1)               
test= test.drop(labels="Fare", axis= 1)                 
test= test.drop(labels="Embarked", axis= 1)              
test.head()

# Definindo se tem Familia
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


#Tratando os Nomes
# Pegando os pronomes de tratamentos
nome= test["Name"]
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

# Colocando variaveis em norminal em numerico
test.loc[test.Sex == "male", 'Sex'] = 0
test.loc[test.Sex == "female", 'Sex'] = 1

# Tratar Valores NaN
print(test.isnull().sum()) 

idade_media= test["Age"].median()   
test.update(test["Age"].fillna(idade_media))

print("\n")
test.isnull().sum()

# Tratando Dados Idade
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
  else:
    test["Age"][i] = 8

print(test["Age"].value_counts())

# armazenamento e eliminar

# Passageiros
passageiro= test.iloc[:,0:1]

# Drop
test= test.drop("PassengerId", axis= 1)

# Poly
test= poly.transform(test)

# Normalização dos dados
test= normalizador.transform(test)

# One Hot Encoder
test= oneHot.transform(test).toarray()

# Predic
teste_final= classificador.predict(test)

# Transformando em dataframe e mudando o nome
teste_final = pd.DataFrame(teste_final)
teste_final.columns= ["Survived1", "Survived2"]
teste_final.head()

# Mudando o criterio
teste_final.loc[teste_final.Survived >= 0.5, 'Survived'] = 1
teste_final.loc[teste_final.Survived < 0.5, 'Survived'] = 0
teste_final.head()


# Trocando o tipo da coluna
teste_final["Survived"]=teste_final.Survived.astype('int64')


# Colocando duas DataFrame em um DataFrame
x= pd.concat([passageiro, teste_final], axis= 1)
x.head()

# Index
x.set_index('PassengerId', inplace=True)
x.head()

# Convertendo em CSV
x.to_csv("RN_crossvalidation_Final.csv", sep=',', encoding='utf-8')
