#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
import pickle
import plotly.express as px

from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')




# In[2]:


df = pd.read_csv('../data/datos_procesados.csv')
df.head()


# In[3]:


df.columns


# In[18]:


import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

def generar_grafica(X_train, y_train, X_test, y_test):
    # rf_1 = RandomForestRegressor()
    # rf_1.fit(X_train, y_train)

    rf_1 = pickle.load(open(f"../modelos_entrenados/ModeloRandomForest1.pk", 'rb'))

    # plotting train
    y_pred_train = rf_1.predict(X_train)
    plt.figure(figsize=(17, 7))
    plt.subplot(1, 2, 1)
    plt.scatter(y_train, y_pred_train, alpha=0.2)
    plt.xlabel('Objetivo (y_train)')
    plt.ylabel('Predicción (y_pred)')
    plt.title('Tiempo de entrenamiento')

    # plotting test error
    y_pred_test = rf_1.predict(X_test)
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_pred_test, alpha=0.2)
    plt.xlabel('Objetivos (y_test)')
    plt.ylabel('Predicción (y_pred_train)')
    plt.title('Tiempo de test')

    return plt


# In[1]:


def distribucion_residual(X_train, y_train, X_test, y_test):
    # Inicializa el modelo RandomForestRegressor
    # rf_model = RandomForestRegressor()
    
    # Entrena el modelo
    # rf_model.fit(X_train, y_train)
    rf_model = pickle.load(open(f"../modelos_entrenados/ModeloRandomForest1.pk", 'rb'))
    
    # Calcula los residuos para los datos de entrenamiento y prueba
    residual_train = y_train - rf_model.predict(X_train)
    residual_test = y_test - rf_model.predict(X_test)

    # Plotting para el conjunto de entrenamiento
    plt.figure(figsize=(17,7))
    plt.subplot(1,2,1)
    sns.histplot(residual_train, kde=True)  # Utiliza histplot en lugar de distplot
    plt.title('Residuos del Entrenamiento')

    # Plotting para el conjunto de prueba
    plt.subplot(1,2,2)
    sns.histplot(residual_test, kde=True)  # Utiliza histplot en lugar de distplot
    plt.title('Residuos del Test')

    # Devuelve el gráfico
    return plt


# In[2]:


def graf_oscilador(X_test, y_test):

    rf_1 = pickle.load(open(f"../modelos_entrenados/ModeloRandomForest1.pk", 'rb'))

    # Crear DataFrame para comparar resultados y predicciones
    df_eval = pd.DataFrame(rf_1.predict(X_test), columns=['Prediccion'])
    # Añadir columna 'Objetivo'
    y_test = y_test.reset_index(drop=True)
    df_eval['Objetivo'] = y_test

    # Calcular columnas 'Residual' y 'Difference%'
    df_eval['Residual'] = df_eval['Objetivo'] - df_eval['Prediccion']
    df_eval['Difference%'] = np.absolute(df_eval['Residual'] / df_eval['Objetivo'] * 100)

    # Visualizar un indicador tipo oscilador
    # X = np.linspace(0, 250, 250)
    # Y1 = df_eval["Prediccion"].head(250).values
    # Y2 = df_eval["Objetivo"].head(250).values

    # plt.figure(figsize=(25, 8))
    # plt.plot(X, Y1, color="blue", label="Predicción")
    # plt.plot(X, Y2, color="red", label="Objetivo")
    # plt.legend()
    # plt.title('Gráfico Oscilador de Predicciones y Objetivos')
    # plt.xlabel('Índice')
    # plt.ylabel('Valores')
    # plt.show()
    X = np.linspace(0, 150, 150)
    Y1 = df_eval["Prediccion"].head(150).values
    Y2 = df_eval["Objetivo"].head(150).values

    valores = np.concatenate((Y1,Y2))


    indice1 =np.full(Y1.shape[0], fill_value="Predicción")
    indice2 =np.full(Y2.shape[0], fill_value="Valor real")
    indices = np.concatenate((indice1,indice2))

    df_comparando = pd.DataFrame({'x': np.concatenate((X,X)),
                    'y': valores,
                    'grupo' : indices})

    fig =  px.line(df_comparando, x = 'x', y = 'y', color = 'grupo',color_discrete_map = {'Predicción':'blue', 'Valorr real':'red'})

    fig.update_layout(
        autosize=False,
        width=1400,
        height=600,
    )
    return fig