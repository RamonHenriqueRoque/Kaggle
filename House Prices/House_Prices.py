# -*- coding: utf-8 -*-
"""
# **Importar bibliotecas basicas**
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import log10, sqrt
from prettytable import PrettyTable

plt.rcParams["figure.figsize"]= 15, 6             # Para aumentar o grafico

import warnings
warnings.filterwarnings('ignore')                 # Eliminar mensagens

!pip install pyod
!pip install lazypredict
!pip install ppscore

"""# **Importar Bases**"""

from sklearn.utils import shuffle                 # Para bagunças a ordem
base= shuffle(pd.read_csv("train.csv"), random_state= 1)
base= base.drop(labels= "Id", axis= 1)
base.head()

"""# **Observar/Analisar as colunas**

**ANALISAR DADOS**
"""

nome_colunas= list(base.columns)                                                # Pegando os nomes das colunas
nome_colunas.pop(0)                                                             # Eliminando o id
bins= int(round(1+3.322*np.log10(len(base)),0))                                 # Regra de sturges
lista_grafico=[]
lista_object=[]

for i in nome_colunas:
  print("Nome da coluna:",i)
  print("Tipo da coluna:", base[i].dtype)

  if base[i].dtype == "int64" or base[i].dtype == "float32" or base[i].dtype == "float64":
    base_limpa= base[i].dropna()
    plt.hist(x=base_limpa, bins= bins)
    plt.show()
    print("\n", "Descrisão estatisticas:")
    print(base[i].describe())
    print("\n", "Descrisão estatisticas:", "\n","Quant.  Valores","\n",base[i].value_counts())
  
  elif base[i].dtype == "object":
    print("Valores da coluna:")
    print(base[i].value_counts())
    lista_object.append(i)
  
  if base[i].isnull().sum() != 0:
    print("Valores Nulos:", base[i].isnull().sum())
  else:
    print("Não Existe valores Nulos.")
  print(3*"\n", 3*"#####################################################")

"""**Grafico**"""

plt.rcParams["figure.figsize"]= 15, 8
for i in lista_object:
  title= "Nome da Coluna: " + i
  legenda= []
  for j in range(len(base[i].value_counts(dropna= False).index)):
    legenda.append([base[i].value_counts(dropna= False).index[j],base[i].value_counts(dropna= False)[j]])
  
  plt.pie(base[i].value_counts(dropna= False),  labels= base[i].value_counts(dropna= False))
  plt.title(title, fontsize= 20)
  plt.legend(labels= base[i].value_counts(dropna= False).index, loc= "best", fontsize= 12)
  plt.show()
  print(4*"#####################")

"""# **Pre-processamento parte 1**

> Arrumar as colunas e eliminar outras.

**Pegando valores NaN**
"""

# Fazendo uma lista das colunas vazias
coluna_nan= []
for i in base.columns:
  nulo= base[i].isnull().sum()
  if nulo != 0:
    coluna_nan.append([i, nulo])

estrutura_tabela= PrettyTable()
estrutura_tabela.field_names= ["INDEX", "NOME DA COLUNA", "QUANT."]

for coluna in coluna_nan:
  estrutura_tabela.add_row([coluna_nan.index(coluna), coluna[0], coluna[1]])

print(estrutura_tabela)

# Separando os valores faltantes
lista_de_valores_faltantes_numerico=[]
lista_de_valores_faltantes_string=[]
for i in coluna_nan:
  try:
    float(base[i[0]].value_counts().index[0])
    lista_de_valores_faltantes_numerico.append(i[0])
  except:
    lista_de_valores_faltantes_string.append(i[0])


# Separandos os dois tipos de NAN no String
lista_de_comodos_que_casa_nao_tem=["Alley", "BsmtQual", "BsmtCond", "FireplaceQu", "BsmtExposure", "BsmtFinType1",
                                   "BsmtFinType2", "GarageType", "GarageFinish", "GarageQual", "GarageCond", 
                                   "PoolQC", "Fence", "MiscFeature"]
                                   
for coluna_remover in lista_de_comodos_que_casa_nao_tem:
  lista_de_valores_faltantes_string.remove(coluna_remover)

# Tratando com os floats
for i in lista_de_valores_faltantes_numerico:
  base.update(base[i].fillna(round(base[i].mean(),2)))

# Tratando com os string
for j in lista_de_comodos_que_casa_nao_tem:         # Quando nao houver comodo.
  base.update(base[j].fillna("There_is_Not"))

"""Quando fizer o tratamento de String para numero, vou terminar os NAN"""

"""**Juntar Criterios**"""

# lotshapePs => IR1, IR2, IR3 => IR
base["LotShape"].loc[base.LotShape == "IR1"] = "IR"
base["LotShape"].loc[base.LotShape == "IR2"] = "IR"
base["LotShape"].loc[base.LotShape == "IR3"] = "IR"

# LandContour => Bnk, Hls, Low
base["LandContour"].loc[base.LandContour == "Bnk"] = "Slope"
base["LandContour"].loc[base.LandContour == "HLS"] = "Slope"
base["LandContour"].loc[base.LandContour == "Low"] = "Slope"

# LotConfig => FR2 e FR3
base["LotConfig"].loc[base.LotConfig == "FR2"] = "FR"
base["LotConfig"].loc[base.LotConfig == "FR3"] = "FR"

# (Ex,Gd) = BOM, (TA)= Regular, (Fa, Po)= Ruim
lista_junta_coluna_criterio= ["ExterQual", "ExterQual", "BsmtQual", "BsmtCond", "HeatingQC",
                              "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC"]

def junta_coluna_criterio(lista):
  for nome_coluna in lista:
    base.loc[base[nome_coluna] == "Ex", nome_coluna]= "Acima_Media"
    base.loc[base[nome_coluna] == "Gd", nome_coluna]= "Acima_Media"
    base.loc[base[nome_coluna] == "TA", nome_coluna]= "Regular"
    base.loc[base[nome_coluna] == "Fa", nome_coluna]= "Abaixo_Media"
    base.loc[base[nome_coluna] == "Po", nome_coluna]= "Abaixo_Media"

junta_coluna_criterio(lista_junta_coluna_criterio)

"""**Junção das Colunas**"""

lista_garage_coluna_string= ["GarageFinish","GarageQual","GarageCond"]
lista_garage_coluna_number= ["GarageCars","GarageArea"]
base["Garage"]= 0
base["Garage_car"]= 0

for j in lista_garage_coluna_string:
   for i in range(len(base[j])):
     if base[j][i] == "Acima_Media" or base[j][i] == "Fin":
       base[j][i]= 3
     elif base[j][i] == "Regular" or base[j][i] == "RFn":
       base[j][i]= 2
     elif base[j][i] == "Abaixo_Media" or base[j][i] == "Unf":
       base[j][i]= 1
     else:
       base[j][i]= 0


for i in range(len(base["Garage"])):
  base["Garage"][i]= base["GarageFinish"][i] * base["GarageQual"][i] * base["GarageCond"][i]
  
  if base["GarageCars"][i] != 0:
    base["Garage_car"][i]= round(base["GarageArea"][i] / base["GarageCars"][i],2)
  else:
    base["Garage_car"][i]= 0

for i in lista_garage_coluna_string+lista_garage_coluna_number:
  base= base.drop(labels= i, axis= 1)

# Bsmt
def arrumar_valores(lista_coluna):
  for j in lista_coluna:
    for i in range(len(base[j])):
      if base[j][i] == "GLQ":
        base[j][i]= 5
      elif base[j][i] == "ALQ":
        base[j][i]= 4
      elif base[j][i] == "BLQ":
        base[j][i]= 3
      elif base[j][i] == "Rec":
        base[j][i]= 2
      elif base[j][i] == "LwQ":
        base[j][i]= 1
      else:
        base[j][i]= 0

arrumar_valores(["BsmtFinType1", "BsmtFinType2"])

base["Bsmt_Bath"]= 0
base["BsmtType1"]= 0
base["BsmtType2"]= 0
base["BsmtIndTermino"]=0
lista_fusao_Bsmt=["BsmtFinType1", "BsmtFinSF1", "BsmtFinType2", "BsmtFinSF2", "BsmtUnfSF",
                  "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath"]

for i in range(len(base)):

  soma_Bsmt2= base["BsmtFinType1"][i] * base["BsmtFinSF1"][i]
  soma_Bsmt3= base["BsmtFinSF2"][i] * base["BsmtFinType2"][i]
  indice_termino= (base["BsmtUnfSF"][i] / base["TotalBsmtSF"][i])* 100
  soma_Bsmt_bath=base["BsmtFullBath"][i] + base["BsmtHalfBath"][i]

  base["BsmtType1"][i]= soma_Bsmt2
  base["BsmtType2"][i]= soma_Bsmt3
  base["BsmtIndTermino"][i]= round(indice_termino,2)
  base["Bsmt_Bath"][i]= soma_Bsmt_bath
  
# Arrumandos os nan do BsmtIndTermino
base["BsmtIndTermino"].update(base["BsmtIndTermino"].fillna(0))

for i in lista_fusao_Bsmt:
  base= base.drop(labels= i, axis= 1)

# Kitchen e KitchenQual
base["Kitchen"]= 0
for i in range(len(base)):
  if base["KitchenQual"][i] == "Acima_Media":
    base["Kitchen"][i]= 2 * base["KitchenAbvGr"][i]
  elif base["KitchenQual"][i] == "Regular":
    base["Kitchen"][i]= 1 * base["KitchenAbvGr"][i]
  elif base["KitchenQual"][i] == "Abaixo_Media":
    base["Kitchen"][i]= 0 * base["KitchenAbvGr"][i]

base= base.drop(columns= ["KitchenAbvGr", "KitchenQual"], axis= 1)

"""**Arrumando Escalas**"""

# Arrumando as colunas
def arrumando_escala(lista_coluna):
  for coluna in lista_coluna:
    bins= int(sqrt(len(list(base[coluna].value_counts(dropna= False).index))))
    lista_escala_data= np.histogram(base[coluna], bins= bins)[1]
    
    for linha in range(len(base)):
      if base[coluna][linha] >= lista_escala_data[0] and base[coluna][linha] <= lista_escala_data[1]:
        base[coluna][linha] = 1
      elif base[coluna][linha] > lista_escala_data[1] and base[coluna][linha] <= lista_escala_data[2]:
        base[coluna][linha] = 2
      elif base[coluna][linha] > lista_escala_data[2] and base[coluna][linha] <= lista_escala_data[3]:
        base[coluna][linha] = 3
      elif base[coluna][linha] > lista_escala_data[3] and base[coluna][linha] <= lista_escala_data[4]:
        base[coluna][linha] = 4
      elif base[coluna][linha] > lista_escala_data[4] and base[coluna][linha] <= lista_escala_data[5]:
        base[coluna][linha] = 5 
      elif base[coluna][linha] > lista_escala_data[5] and base[coluna][linha] <= lista_escala_data[6]:
        base[coluna][linha] = 6
      elif base[coluna][linha] > lista_escala_data[6] and base[coluna][linha] <= lista_escala_data[7]:
        base[coluna][linha] = 7
      elif base[coluna][linha] > lista_escala_data[7] and base[coluna][linha] <= lista_escala_data[8]:
        base[coluna][linha] = 8
      elif base[coluna][linha] > lista_escala_data[8] and base[coluna][linha] <= lista_escala_data[9]:
        base[coluna][linha] = 9
      elif base[coluna][linha] > lista_escala_data[9] and base[coluna][linha] <= lista_escala_data[10]:
        base[coluna][linha] = 10
      else:
        base[coluna][linha] =11

lista_para_arrumar_escala= ["YrSold", "YearBuilt","YearRemodAdd", "GarageYrBlt"]
arrumando_escala(lista_para_arrumar_escala)

# MSSubClass
valores_MSSubClass= sorted(list(base["MSSubClass"].value_counts().index))

for i in range(len(base)):
  for j in valores_MSSubClass:
    if base["MSSubClass"][i] == j:
      base["MSSubClass"][i]= valores_MSSubClass.index(j)

"""**Pegando as colunas Discretas**"""

#Pegando os valores categorico
coluna_discreta_string= []
for i in base.columns:
  if base[i].dtype == "object":
    coluna_discreta_string.append(i)

coluna_discreta_number= ["MSSubClass", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd",
                         "FullBath", "HalfBath", "BedroomAbvGr",	"TotRmsAbvGrd",
                         "Fireplaces",	"GarageYrBlt",  "Bsmt_Bath", "Kitchen", "YrSold"]

coluna_discreta_total= coluna_discreta_number + coluna_discreta_string

"""**Transformar categorico em discreto**"""

biblioteca_categorico_discretos= {}

# Colocando os nomes das chaves
for nome_coluna in coluna_discreta_string:
  biblioteca_categorico_discretos[nome_coluna]= []

# Colocando os valores
for valor in coluna_discreta_string:
  lista_categorico= base[valor].value_counts(dropna= False).index

  if "There_is_Not" in lista_categorico:
    for categorico in lista_categorico:
      biblioteca_categorico_discretos[valor].append(categorico)
      
  else:
    biblioteca_categorico_discretos[valor].append("There_is_Not")
    for categorico in lista_categorico:
      biblioteca_categorico_discretos[valor].append(categorico)

# Transformando categorico em discreto
for coluna in coluna_discreta_string:
  for linha in range(len(base)):
    if base[coluna][linha] != np.nan:
      nome_linha_categorica= base[coluna][linha]
      base[coluna][linha]= list(biblioteca_categorico_discretos[coluna]).index(nome_linha_categorica)

base.head()

"""**Arrumando o restos dos NaN**"""

# Eliminando os valores na base de NaN, caso eu tenha excluido.
for coluna in lista_de_valores_faltantes_string:
  if coluna not in list(base.columns):
    lista_de_valores_faltantes_string.remove(coluna)

# Tratando com os floats
for i in lista_de_valores_faltantes_string:
  if i in base.columns:
    base.update(base[i].fillna(round(base[i].median(),2)))

"""# **Estatisticas**"""

base_estatistica= base.drop(columns= coluna_discreta_total, axis=1)

"""**Plotar**"""

print("\t\t\tGrafico de dispersão")
sns.pairplot(base_estatistica)
plt.show()

"""**Correlação**"""

# Correlação.

correlacao= base.corr().abs()                        # Pego os valores de correlação

sns.set(rc={'figure.figsize':(20,15)})         # Tamanho da imagem
ax = sns.heatmap(correlacao,                   # Base
                 annot = True,                 # Mapa de Calor
                 annot_kws = {'size': 8},      # tamnho do texto
                 fmt = '.1f',                  # quantas casas decimais
                 cmap = 'PiYG',                # Definindo cor
                 linewidths = 1,               # coluna de escala de cor
                )

# Pegando as correlações maiores que 0.65
lista_correlacao_eliminar=[]
for i in correlacao.index:            # ID
  for j in correlacao.columns:         # Colunas
    if correlacao[j][i] >= 0.65 and correlacao[j][i] < 1:
      lista_correlacao_eliminar.append(i)

# Pegando valores sem repetir
lista_correlacao_eliminar= set(lista_correlacao_eliminar)
lista_correlacao_eliminar.remove("SalePrice")
print(lista_correlacao_eliminar)

# Eliminando as colunas
for i in lista_correlacao_eliminar:
  base= base.drop(labels= i, axis=1)

"""**PPS**"""

import ppscore 

pps= ppscore.matrix(base)
pps = pps[["x", "y", "ppscore"]].pivot(columns="x", index="y", values="ppscore")

sns.set(rc={'figure.figsize':(20,15)})         # Tamanho da imagem
ax = sns.heatmap(pps,                          # Base
                 annot = True,                 # Mapa de Calor
                 annot_kws = {'size': 8},      # tamnho do texto
                 fmt = '.1f',                  # quantas casas decimais
                 cmap = 'YlGnBu',              # Definindo cor
                 linewidths = 1,               # coluna de escala de cor
                )

# Pegando as correlações maiores que 0.65
lista_pps_eliminar=[]
for i in pps.index:            # ID
  for j in pps.columns:         # Colunas
    if pps[j][i] >= 0.65 and pps[j][i] < 1:
      lista_pps_eliminar.append(i)

# Pegando valores sem repetir
lista_pps_eliminar= set(lista_pps_eliminar)
try:
  lista_pps_eliminar.remove("SalePrice")
except:
  pass
  
# Eliminando as colunas
for i in lista_pps_eliminar:
  base= base.drop(labels= i, axis=1)

"""# **Pre-processamento parte 2**

> Com os dados ja preparados, usar medotos para melhorar os dados em reação ao algoritmo

**Regressor e preço**
"""

previsores= base.drop(labels= "SalePrice", axis= 1)
preco= base["SalePrice"]
previsores.head()

"""**LOGe(x+1)**"""

for coluna in previsores.columns:
  for linha in previsores.index:
    previsores[coluna][linha]= np.log1p(previsores[coluna][linha])
previsores.head()

"""**OUTLIERS usando a biblioteca PyOD**

Auto Encoder
"""

from pyod.models.auto_encoder import AutoEncoder

outliers_AutoEncoder= AutoEncoder(epochs= 250, random_state= 4).fit_predict(previsores, preco)

index_outliers_AutoEncoder=[]

for linhas in range(len(base)):
  if outliers_AutoEncoder[linhas] == 1:
    index_outliers_AutoEncoder.append(linhas)

previsores= previsores.drop(labels= index_outliers_AutoEncoder, axis=0)
preco= preco.drop(labels= index_outliers_AutoEncoder, axis=0)

# ARRUMAR O INDEX
previsores.index, preco.index= [i for i in range(len(previsores))], [i for i in range(len(previsores))]

print(previsores.shape)

"""**OUTLIERS com o IsolationForest**"""

from sklearn.ensemble import IsolationForest
outliers_isolationForest= IsolationForest().fit_predict(base)

index_outliers_isolationforest=[]

for linhas in range(len(base)):
  if outliers_isolationForest[linhas] == -1:
    index_outliers_isolationforest.append(linhas)

previsores= previsores.drop(labels= index_outliers_isolationforest, axis=0)
preco= preco.drop(labels= index_outliers_isolationforest, axis=0)

# ARRUMAR O INDEX
previsores.index, preco.index= [i for i in range(len(previsores))], [i for i in range(len(previsores))]

"""**Escolhendo colunas com o XGBOOST**"""

from xgboost import XGBRegressor, plot_importance
modelo_coluna_xgboost= XGBRegressor(importance_type= "weight", random_state=4).fit(np.array(previsores), np.array(preco))

plot_importance(modelo_coluna_xgboost, grid= False, max_num_features= 20, title= "Feature importance TOP 20")
plt.show()

# PEGANDO AS COLUNAS;
lista_escolha_coluna_xgboost= []
lista_colunas_eliminadas_xgboost= []
criterio_escolha_coluna_xgboost= 0.0

## 
for i in range(len(modelo_coluna_xgboost.feature_importances_)):
  lista_escolha_coluna_xgboost.append([modelo_coluna_xgboost.feature_importances_[i], previsores.columns[i]])

lista_escolha_coluna_xgboost= sorted(lista_escolha_coluna_xgboost, reverse =True)

for valor_nome in lista_escolha_coluna_xgboost:
  if valor_nome[0] <= criterio_escolha_coluna_xgboost: 
    previsores= previsores.drop(labels= valor_nome[1], axis= 1)
    lista_colunas_eliminadas_xgboost.append(valor_nome[1])
print(previsores.shape)

"""**Redução de dimensão**"""

teste_dimensao= np.copy(previsores)
teste_dimensao= np.array(teste_dimensao)
teste_dimensao

from sklearn.decomposition import PCA
pca = PCA(n_components = None)
pca.fit(teste_dimensao)

print((pca.explained_variance_ratio_))
print("Usando apens tres dimensões temos {}%".format(pca.explained_variance_ratio_[0]+pca.explained_variance_ratio_[1]+pca.explained_variance_ratio_[3]))

# Caso queira diminuir as dimensoes
pca_oficial= PCA(n_components= 3)
previsores= pd.DataFrame(pca_oficial.fit_transform(np.array(previsores)))
previsores.head()

"""**Normalização**"""

#Normalização dos Dados
from sklearn.preprocessing import StandardScaler
scaler_previsores= StandardScaler()
previsores= scaler_previsores.fit_transform(previsores)

for i in range(len(previsores)):
  previsores[i]=previsores[i].astype('float64')

previsores= pd.DataFrame(previsores)
previsores.head()

"""# **Algorimos**

**Auto-ML de Algoritmo Teste**
"""

from lazypredict.Supervised import LazyRegressor
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test= train_test_split(previsores, preco, test_size= 0.1, random_state= 1)

lazy= LazyRegressor(verbose=0,ignore_warnings=False, custom_metric=None )

modelos_lazy, predicoes_lazy= lazy.fit(x_train, x_test, y_train, y_test)
modelos_lazy

"""**ALGORITMO PARA TURNING**"""

# GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

# Modelo gradientBoosting
regressor_gradientBoosting= GradientBoostingRegressor(max_depth= 5, min_samples_leaf= 2, min_samples_split= 100, 
                                                      random_state= 4, learning_rate= 0.04, n_estimators= 1400)

# Criando parametros
parametros_gradientBoosting= {}       


# criando
greidSearch_gradientBoosting= GridSearchCV(estimator= regressor_gradientBoosting,
                                           param_grid= parametros_gradientBoosting,         
                                           scoring= "neg_root_mean_squared_error",                         
                                           cv=10,
                                           n_jobs= -1)                                       
                          
# Treinamento
greidSearch_gradientBoosting=greidSearch_gradientBoosting.fit(previsores, preco)

# Resultados
from sklearn.metrics import mean_squared_error
melhores_paramentros= greidSearch_gradientBoosting.best_params_
melhor_precisao= greidSearch_gradientBoosting.best_score_

print("######"*10, " \nMelhores parametros foram:", melhores_paramentros)
print("A media de Precisao foi de:", abs(melhor_precisao))
print("RMSE Sem Normalização:", mean_squared_error(greidSearch_gradientBoosting.predict(previsores), preco ,squared= False))

# Observando os resultados dos CVS
print("######"*10)
for i in range(len(list(greidSearch_gradientBoosting.cv_results_.values())[0])):       # Numero das combinaçoes
  for nome_dos_resultados in list(greidSearch_gradientBoosting.cv_results_.keys()):    # Nomes das chaves/key 
    if "time" not in nome_dos_resultados.split("_") and greidSearch_gradientBoosting.cv_results_["rank_test_score"][i]  <= 10 :                                # Nesse projeto, não estou me preocupando com o tempo
      print(nome_dos_resultados, "=", greidSearch_gradientBoosting.cv_results_[nome_dos_resultados][i])

"""**ALGORITMO OFICIAL**"""

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Modelo gradientBoosting
regressor_gradientBoosting= GradientBoostingRegressor(learning_rate= 0.01, n_estimators=1500, max_depth= 4,
                                                      min_samples_leaf= 3, min_samples_split= 10, warm_start= False, 
                                                      random_state= 4, verbose= 0).fit(previsores, preco)

print("RMSE Sem Normalização:", mean_squared_error(regressor_gradientBoosting.predict(previsores), preco,squared= False))

"""**PLOTAR TREE**"""

from sklearn.tree import export_graphviz
from IPython.display import Image

export_graphviz(regressor_gradientBoosting.estimators_[0, 0], out_file='gradientBoosting.dot',
                feature_names = list(previsores.columns), class_names = ["SalePrice"],
                filled=True, rounded=True, leaves_parallel= True, precision =2, impurity=False)

!dot -Tpng gradientBoosting.dot -o gradientBoosting.png -Gdpi=500
Image("gradientBoosting.png", width= 1920)

"""**GRAVAR E CARREGAR MODELO**"""

# GRAVAR MODELO
import pickle
pickle.dump(regressor_gradientBoosting, open("GradientBoosting_PriceHouse.sav", "wb"))

# Carregar Modelo
regressor= pickle.load(open("GradientBoosting_PriceHouse.sav", "rb"))

"""# **Kaggle Pre-processamento parte 1**

> Arrumar as colunas e eliminar outras.

**Carregar Base**
"""

teste= pd.read_csv("test.csv")

"""**Pegando valores NaN**"""

# Fazendo uma lista das colunas vazias
coluna_nan= []
for i in teste.columns:
  nulo= teste[i].isnull().sum()
  if nulo != 0:
    coluna_nan.append([i, nulo])

estrutura_tabela= PrettyTable()
estrutura_tabela.field_names= ["INDEX", "NOME DA COLUNA", "QUANT."]

for coluna in coluna_nan:
  estrutura_tabela.add_row([coluna_nan.index(coluna), coluna[0], coluna[1]])

print(estrutura_tabela)

# Separando os valores faltantes
lista_de_valores_faltantes_numerico=[]
lista_de_valores_faltantes_string=[]
for i in coluna_nan:
  try:
    float(teste[i[0]].value_counts().index[0])
    lista_de_valores_faltantes_numerico.append(i[0])
  except:
    lista_de_valores_faltantes_string.append(i[0])


# Separandos os dois tipos de NAN no String
lista_de_comodos_que_casa_nao_tem=["Alley", "BsmtQual", "BsmtCond", "FireplaceQu", "BsmtExposure", "BsmtFinType1",
                                   "BsmtFinType2", "GarageType", "GarageFinish", "GarageQual", "GarageCond", 
                                   "PoolQC", "Fence", "MiscFeature"]
                                   
for coluna_remover in lista_de_comodos_que_casa_nao_tem:
  lista_de_valores_faltantes_string.remove(coluna_remover)

# Tratando com os floats
for i in lista_de_valores_faltantes_numerico:
  teste.update(teste[i].fillna(round(teste[i].mean(),2)))

# Tratando com os string
for j in lista_de_comodos_que_casa_nao_tem:         # Quando nao houver comodo.
  teste.update(teste[j].fillna("There_is_Not"))

"""Quando fizer o tratamento de String para numero, vou terminar os NAN"""

"""**Juntar Criterios**"""

# lotshapePs => IR1, IR2, IR3 => IR
teste["LotShape"].loc[teste.LotShape == "IR1"] = "IR"
teste["LotShape"].loc[teste.LotShape == "IR2"] = "IR"
teste["LotShape"].loc[teste.LotShape == "IR3"] = "IR"

# LandContour => Bnk, Hls, Low
teste["LandContour"].loc[teste.LandContour == "Bnk"] = "Slope"
teste["LandContour"].loc[teste.LandContour == "HLS"] = "Slope"
teste["LandContour"].loc[teste.LandContour == "Low"] = "Slope"

# LotConfig => FR2 e FR3
teste["LotConfig"].loc[teste.LotConfig == "FR2"] = "FR"
teste["LotConfig"].loc[teste.LotConfig == "FR3"] = "FR"

# (Ex,Gd) = BOM, (TA)= Regular, (Fa, Po)= Ruim
lista_junta_coluna_criterio= ["ExterQual", "ExterQual", "BsmtQual", "BsmtCond", "HeatingQC",
                              "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC"]

def junta_coluna_criterio(lista):
  for nome_coluna in lista:
    teste.loc[teste[nome_coluna] == "Ex", nome_coluna]= "Acima_Media"
    teste.loc[teste[nome_coluna] == "Gd", nome_coluna]= "Acima_Media"
    teste.loc[teste[nome_coluna] == "TA", nome_coluna]= "Regular"
    teste.loc[teste[nome_coluna] == "Fa", nome_coluna]= "Abaixo_Media"
    teste.loc[teste[nome_coluna] == "Po", nome_coluna]= "Abaixo_Media"

junta_coluna_criterio(lista_junta_coluna_criterio)

"""**Junção das Colunas**"""

lista_garage_coluna_string= ["GarageFinish","GarageQual","GarageCond"]
lista_garage_coluna_number= ["GarageCars","GarageArea"]
teste["Garage"]= 0
teste["Garage_car"]= 0

for j in lista_garage_coluna_string:
   for i in range(len(teste[j])):
     if teste[j][i] == "Acima_Media" or teste[j][i] == "Fin":
       teste[j][i]= 3
     elif teste[j][i] == "Regular" or teste[j][i] == "RFn":
       teste[j][i]= 2
     elif teste[j][i] == "Abaixo_Media" or teste[j][i] == "Unf":
       teste[j][i]= 1
     else:
       teste[j][i]= 0


for i in range(len(teste["Garage"])):
  teste["Garage"][i]= teste["GarageFinish"][i] * teste["GarageQual"][i] * teste["GarageCond"][i]
  
  if teste["GarageCars"][i] != 0:
    teste["Garage_car"][i]= round(teste["GarageArea"][i] / teste["GarageCars"][i],2)
  else:
    teste["Garage_car"][i]= 0

for i in lista_garage_coluna_string+lista_garage_coluna_number:
  teste= teste.drop(labels= i, axis= 1)

# Bsmt
def arrumar_valores(lista_coluna):
  for j in lista_coluna:
    for i in range(len(teste[j])):
      if teste[j][i] == "GLQ":
        teste[j][i]= 5
      elif teste[j][i] == "ALQ":
        teste[j][i]= 4
      elif teste[j][i] == "BLQ":
        teste[j][i]= 3
      elif teste[j][i] == "Rec":
        teste[j][i]= 2
      elif teste[j][i] == "LwQ":
        teste[j][i]= 1
      else:
        teste[j][i]= 0

arrumar_valores(["BsmtFinType1", "BsmtFinType2"])

teste["Bsmt_Bath"]= 0
teste["BsmtType1"]= 0
teste["BsmtType2"]= 0
teste["BsmtIndTermino"]=0
lista_fusao_Bsmt=["BsmtFinType1", "BsmtFinSF1", "BsmtFinType2", "BsmtFinSF2", "BsmtUnfSF",
                  "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath"]

for i in range(len(teste)):

  soma_Bsmt2= teste["BsmtFinType1"][i] * teste["BsmtFinSF1"][i]
  soma_Bsmt3= teste["BsmtFinSF2"][i] * teste["BsmtFinType2"][i]
  indice_termino= (teste["BsmtUnfSF"][i] / teste["TotalBsmtSF"][i])* 100
  soma_Bsmt_bath=teste["BsmtFullBath"][i] + teste["BsmtHalfBath"][i]

  teste["BsmtType1"][i]= soma_Bsmt2
  teste["BsmtType2"][i]= soma_Bsmt3
  teste["BsmtIndTermino"][i]= round(indice_termino,2)
  teste["Bsmt_Bath"][i]= soma_Bsmt_bath
  
# Arrumandos os nan do BsmtIndTermino
teste["BsmtIndTermino"].update(teste["BsmtIndTermino"].fillna(0))

for i in lista_fusao_Bsmt:
  teste= teste.drop(labels= i, axis= 1)

# Kitchen e KitchenQual
teste["Kitchen"]= 0
for i in range(len(teste)):
  if teste["KitchenQual"][i] == "Acima_Media":
    teste["Kitchen"][i]= 2 * teste["KitchenAbvGr"][i]
  elif teste["KitchenQual"][i] == "Regular":
    teste["Kitchen"][i]= 1 * teste["KitchenAbvGr"][i]
  elif teste["KitchenQual"][i] == "Abaixo_Media":
    teste["Kitchen"][i]= 0 * teste["KitchenAbvGr"][i]

teste= teste.drop(columns= ["KitchenAbvGr", "KitchenQual"], axis= 1)

"""**Arrumando Escalas**"""

# Arrumando as colunas
def arrumando_escala(lista_coluna):
  for coluna in lista_coluna:
    bins= int(sqrt(len(list(teste[coluna].value_counts(dropna= False).index))))
    lista_escala_data= np.histogram(teste[coluna], bins= bins)[1]
    
    for linha in range(len(teste)):
      if teste[coluna][linha] >= lista_escala_data[0] and teste[coluna][linha] <= lista_escala_data[1]:
        teste[coluna][linha] = 1
      elif teste[coluna][linha] > lista_escala_data[1] and teste[coluna][linha] <= lista_escala_data[2]:
        teste[coluna][linha] = 2
      elif teste[coluna][linha] > lista_escala_data[2] and teste[coluna][linha] <= lista_escala_data[3]:
        teste[coluna][linha] = 3
      elif teste[coluna][linha] > lista_escala_data[3] and teste[coluna][linha] <= lista_escala_data[4]:
        teste[coluna][linha] = 4
      elif teste[coluna][linha] > lista_escala_data[4] and teste[coluna][linha] <= lista_escala_data[5]:
        teste[coluna][linha] = 5 
      elif teste[coluna][linha] > lista_escala_data[5] and teste[coluna][linha] <= lista_escala_data[6]:
        teste[coluna][linha] = 6
      elif teste[coluna][linha] > lista_escala_data[6] and teste[coluna][linha] <= lista_escala_data[7]:
        teste[coluna][linha] = 7
      elif teste[coluna][linha] > lista_escala_data[7] and teste[coluna][linha] <= lista_escala_data[8]:
        teste[coluna][linha] = 8
      elif teste[coluna][linha] > lista_escala_data[8] and teste[coluna][linha] <= lista_escala_data[9]:
        teste[coluna][linha] = 9
      elif teste[coluna][linha] > lista_escala_data[9] and teste[coluna][linha] <= lista_escala_data[10]:
        teste[coluna][linha] = 10
      else:
        teste[coluna][linha] =11

lista_para_arrumar_escala= ["YrSold", "YearBuilt","YearRemodAdd", "GarageYrBlt"]
arrumando_escala(lista_para_arrumar_escala)

# MSSubClass
valores_MSSubClass= sorted(list(teste["MSSubClass"].value_counts().index))

for i in range(len(teste)):
  for j in valores_MSSubClass:
    if teste["MSSubClass"][i] == j:
      teste["MSSubClass"][i]= valores_MSSubClass.index(j)

"""**Pegando as colunas Discretas**"""

#Pegando os valores categorico
coluna_discreta_string= []
for i in teste.columns:
  if teste[i].dtype == "object":
    coluna_discreta_string.append(i)

coluna_discreta_number= ["MSSubClass", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd",
                         "FullBath", "HalfBath", "BedroomAbvGr",	"TotRmsAbvGrd",
                         "Fireplaces",	"GarageYrBlt",  "Bsmt_Bath", "Kitchen", "YrSold"]

coluna_discreta_total= coluna_discreta_number + coluna_discreta_string

"""**Transformar categorico em discreto**"""

biblioteca_categorico_discretos= {}

# Colocando os nomes das chaves
for nome_coluna in coluna_discreta_string:
  biblioteca_categorico_discretos[nome_coluna]= []

# Colocando os valores
for valor in coluna_discreta_string:
  lista_categorico= teste[valor].value_counts(dropna= False).index

  if "There_is_Not" in lista_categorico:
    for categorico in lista_categorico:
      biblioteca_categorico_discretos[valor].append(categorico)
      
  else:
    biblioteca_categorico_discretos[valor].append("There_is_Not")
    for categorico in lista_categorico:
      biblioteca_categorico_discretos[valor].append(categorico)

# Transformando categorico em discreto
for coluna in coluna_discreta_string:
  for linha in range(len(teste)):
    if teste[coluna][linha] != np.nan:
      nome_linha_categorica= teste[coluna][linha]
      teste[coluna][linha]= list(biblioteca_categorico_discretos[coluna]).index(nome_linha_categorica)

teste.head()

"""**Arrumando o restos dos NaN**"""

# Eliminando os valores na teste de NaN, caso eu tenha excluido.
for coluna in lista_de_valores_faltantes_string:
  if coluna not in list(teste.columns):
    lista_de_valores_faltantes_string.remove(coluna)

# Tratando com os floats
for i in lista_de_valores_faltantes_string:
  if i in teste.columns:
    teste.update(teste[i].fillna(round(teste[i].median(),2)))

"""**APAGANDO O ID**"""

# pegando o ID
id= teste["Id"]
teste= teste.drop(labels= "Id", axis= 1)

"""# **Pre-Processamento parte 2**

**Normalização**
"""

#Normalização dos Dados
teste= scaler_previsores.transform(teste)

teste= pd.DataFrame(teste)
teste.head()

"""**Redução de dimensão**"""

# Caso queira diminuir as dimensoes
pca_oficial= PCA(n_components= 3)
previsores= pd.DataFrame(pca_oficial.predict(np.array(previsores)))
previsores.head()

"""**LOGe(x+1)**"""

for coluna in teste.columns:
  for linha in teste.index:
    teste[coluna][linha]= np.log1p(teste[coluna][linha])
teste.head()

"""**PPS**"""

# Eliminando as colunas
teste= teste.drop(columns= lista_pps_eliminar, axis=1)

"""**CORRELAÇÃO LINEAR**"""

# Eliminando as colunas
teste= teste.drop(columns= lista_pps_eliminar, axis=1)

"""**Escolhendo colunas com o XGBOOST**"""

teste= teste.drop(columns= lista_colunas_eliminadas_xgboost, axis= 1)

"""# **Predict**"""

# Predic
teste_final= regressor.predict(teste)


# Transformando em dataframe e mudando o nome
teste_final = pd.DataFrame(teste_final)
teste_final.columns= ["SalePrice"]
teste_final.head()

# Colocando duas DataFrame em um DataFrame
x= pd.concat([id, teste_final], axis= 1)
x.head()

# Index
x.set_index('Id', inplace=True)
x.head()
