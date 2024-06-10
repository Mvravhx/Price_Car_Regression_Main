import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
import seaborn as sns
from RandomForest import graf_oscilador, distribucion_residual, generar_grafica
from sklearn.ensemble import RandomForestRegressor
import joblib

def ml_app():
    st.subheader("Machine Learning Model: Car Price")

    st.markdown("""En la sección de *Exploración de los Datos de Análisis (EDA)* concluimos que el precio tiene una relación lineal 
                con nuestra información. Por lo tanto, procederemos a aplicar diferentes modelos y llegamos a la conclusión, a través 
                de diversas métricas, de que el modelo más eficiente es el Random Forest.""")

    archivo_pkl = 'datos_entrenamiento.pkl'
    contenido = joblib.load(archivo_pkl)

    st.markdown("""Utilice el menú lateral para probar nuestros modelos. Hemos construido un modelo para cada tipo de *Fuel Type*.""")

    st.markdown("""**¿Qué es un modelo de Correlación Lineal?**""")
    st.text("""
            Un modelo de regresión lineal es un enfoque estadístico para modelar la relación entre una variable dependiente 
            (también conocida como variable respuesta o variable a predecir) y una o más variables independientes 
            (también conocidas como variables explicativas o características). El objetivo principal de la regresión lineal 
            es encontrar la mejor línea recta que se ajuste a los datos observados y que pueda utilizarse para predecir valores 
            de la variable dependiente para nuevos datos.""")

    st.markdown("""**¿Qué es un modelo de Random Forest Regressor?**""")
    st.text("""
            El Random Forest Regressor es una técnica de conjunto que combina múltiples árboles de decisión para realizar predicciones. 
            Cada árbol de decisión se entrena con una muestra aleatoria de los datos de entrenamiento, y las predicciones finales se 
            obtienen promediando las predicciones de cada árbol (en el caso de regresión).""")

    st.text("""
            Una vez explicado lo que es un modelo de regresión lineal y especificando cual es el que vamos a utilizar y como se lleva a 
            cabo, procedemos a incluir los siguientes gráficos:""")

    graficos = ["Gráfico de comparación entre predicciones y valores objetivos", 
                "Gráfico de distribución residual", 
                "Gráfico oscilador de predicciones y objetivos"]
    
    st.write(graficos)

    st.text("""
            El gráfico de comparación entre predicciones y valores es un gráfico cuya representación mediante puntos debería construir 
            una línea cuya pendiente sea igual a 1. En este caso podemos ver que los valores se acercan mucho a la realización de la 
            misma debido a que el modelo posee una gran precisión y es bastante acertado. En el caso de que este modelo no fuese tan 
            efectivo los valores se alejarían de la realidad""")

    X_train = contenido['X_train']
    X_test = contenido['X_test']
    y_train = contenido['y_train']
    y_test = contenido['y_test']

    com_pred = generar_grafica(X_train, y_train, X_test, y_test)
    st.pyplot(com_pred)

    st.text("""
            Un gráfico de distribución residual es una herramienta comúnmente utilizada en la evaluación de modelos de regresión para 
            analizar la distribución de los residuos, es decir, las diferencias entre los valores observados y los valores predichos 
            por el modelo.""")

    dist_residual = distribucion_residual(X_train, y_train, X_test, y_test)
    st.pyplot(dist_residual)

    st.text("""
            El gráfico oscilador de predicciones y objetivos es una herramienta de visualización comúnmente utilizada en la evaluación 
            de modelos de regresión, especialmente en series temporales. Este gráfico compara las predicciones del modelo con los valores 
            observados (objetivos) a lo largo del tiempo, mostrando cómo el modelo se desempeña en la tarea de predecir el comportamiento 
            de la serie temporal.""")

    grafica = graf_oscilador(X_test, y_test)
    st.plotly_chart(grafica)

if __name__ == '__main__':
    ml_app()






