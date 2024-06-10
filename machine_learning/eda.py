import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import os
import seaborn as sns
import folium
import matplotlib.patches as patches 
import plotly.graph_objects as go
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import plotly.figure_factory as ff
import pickle

def eda_app():
       
    
    # Variable de estado para controlar qué sección se muestra
    section = st.sidebar.radio("Secciones", ["Dataset", "Histograma", "Gráficos de densidad", "Boxplot", "Sunburst: Relación de carroceria con etiqueta medioambiental & precio", "Barplot: Precio Promedio", "¿Influye el color en el precio?", "Heatmap: Variables con Mayor Correlación","Feature Importance","Conclusión"])
    df = pd.read_csv('../data/datos_limpios.csv')

    # Filtrar y limpiar los datos
    df = df[df["Año"] > 1900]
    df = df[df["KM"] >= 0]
    df = df[df["Precio"] < 500000]
    df = df[df["Aceleracion"] < 30]
    df = df[df["Potencia"] > 0]
    df = df[df["Largo"] > 0]
    df = df[df["Ancho"] > 0]
    df = df[df["Alto"] > 0]
    #KM quitar los de mas de 800k
    # Cambie los valores mayores de 800000 a 800000
    df.loc[df['KM'] > 800000, 'KM'] = 800000
    #Precio borrar el max ya que es un precio demasiado alto y distinto al resto
    df = df.loc[df['Precio'] != 150000000]
    #aceleracion quitar los 0 y pasarlos a 10 y los 100 a 10
    df['Aceleracion'] = np.where(df['Aceleracion'] == 0, 10, np.where(df['Aceleracion'] > 50, 12, df['Aceleracion']))
    #año 0 a 2022
    df['Año'].replace(0, 2022, inplace=True)
    df['Año'].replace(19, 2019, inplace=True)
    #puertas 0 pasarlas a 4
    df['Nº puertas'].replace(0, 4, inplace=True)
    df['Nº puertas'].replace(8, 5, inplace=True)
    df['Nº puertas'].replace(7, 5, inplace=True)
    #potencia 0 a 100 y valores pequeños que no son reales
    df['Potencia'] = df['Potencia'].where(df['Potencia'] != 0, 100)
    filas_con_30 = df['Potencia'] == 30
    df = df[~filas_con_30]
    filas_con_44 = df['Potencia'] == 44
    df = df[~filas_con_44]
    #vel max 0 a 170  otro valor bajo a 180
    # Selecciona las filas donde "Velocidad máxima" es menor que 100
    filas_menores_100 = df['Velocidad máxima'] < 100
    # Actualiza los valores de "Velocidad máxima" a 100 en las filas seleccionadas
    df.loc[filas_menores_100, 'Velocidad máxima'] = 130
    # Cambie los valores negativos a 10
    df.loc[df['KM'] < 0, 'KM'] = 10
    df["KM_log"] = np.log(df["KM"] + 1)
    df["Precio_log"] = np.log(df["Precio"] + 1)   
    
    
    # Dataset
    if section == "Dataset":
        titulo = "<h2 style='color:CornflowerBlue;'>El dataset utilizado detalla los datos de publicación de mas de 52.000 coches:</h2>"
        st.write(titulo, unsafe_allow_html=True)
        st.write("Este dataset cuenta con, además del precio, datos de 21 variables, entre ellas kilometraje, año de venta, ciudad, marca, consumo, etc., obtenidos através del web scrapping la página km77.\n \n" "Con estos datos, nuestro modelo de predicción te dará un precio estimativo de un coche, para que tomes una decisión informada tanto al momento de vender como de comprar tu coche.\n \n" "En las secciones posteriores mostraremos parte del analisis que llevamos a cabo para determinar como mejor seleecionar y trabajar con los datos. Por el momento, aquí puedes hechar un vistazo al dataset:")
        st.write(df)
    
    # Histograma
    elif section == "Histograma":
        # Contenido relacionado con el Histograma
        titulo = "<h2 style='color:CornflowerBlue;'>Histograma:</h2>"
        st.write(titulo, unsafe_allow_html=True)
        st.write("Es una representación gráfica que muestra la distribución de cantidad de un conjunto de datos numéricos mediante barras verticales."
                 "Cada barra representa un intervalo de valores y su altura indica la cantidad con la que los valores caen dentro de ese intervalo. \n\n"
                 "Al observas las barras de un histograma, podemos visualizar patrones, tendencias y valores atípicos en los datos, así como para comparar la distribución entre diferentes conjuntos de datos. \n\n" 
                 "Además, al comparar histogramas de diferentes datos podemos observar si hay correlación en la distribución de estos, teniendo así un mejor entendimiento de las similitudes y diferencias en el comportamiento de las variables en su conjunto. \n\n"
                 "Hemos realizado histogramas de todas las variables con las que contábamos, para poder visualizar patrones y descubrir tendencias. \n\n" 
                 "Abajo puedes visualizar los histogramas de algunas de las variables que evaluamos. \n\n"
                 "Notarás, por ejemplo, que KM, que es el kilometraje de los coches ofertados, muestra coches con valores de hasta 800.000 km, sin embargo, estos son valores excepcionales, siendo la mayoría de las ofertas de hasta 200.000 km. \n\n"
                 "Si observas el histograma de Combustibles, verás que la gran mayoría de las ofertas se concentran en choches Diésel y Gasolina. Si pasas ahora el histograma de Etiqueta medioambiental, verás que la mayoría de los coches con de categoría C. Podríamos intuir entonces una correlación entre estás dos variables. Sin embargo, si pasas ahora a al histograma de Año, verás que la mayoría de las ofertas son de autos de hasta 10 años, por lo cual vemos que aún con la introducción de nuevos tipos de combustibles, la gasolina y el diésel, con menor categoría medioambiental, siguen siendo los de mayor circulación en el mercado. \n\n")
        st.markdown("---")
        
        # Seleccionar columna para el histograma
        selected_column = st.selectbox("Columna:", ["KM", "Año", "Combustible", "Potencia", "Velocidad máxima", "Aceleracion", "Consumo", "Etiqueta medioambiental", "Precio", "Largo", "Precio_log"])
        
        # Generar histograma al hacer clic en el botón
        if st.button("Generar Histograma"):
             if selected_column:
                st.write(f"Histograma de la columna '{selected_column}':")
                fig, ax = plt.subplots()
                fig = px.histogram(df, x=selected_column, nbins=50)
                fig.update_layout(autosize=False, width=900, height=600)
            
                ax.set_title(selected_column)
                ax.set_xlabel("Valor")
                ax.set_ylabel("Cantidad de coches")
                st.plotly_chart(fig)
            
                # Mostrar explicación si se selecciona la columna "Precio_log"
                if selected_column == "Precio_log":
                    st.write("El precio logarítmico es una transformación aplicada al precio original utilizando el logaritmo natural. "
                         "Esta transformación se utiliza para estabilizar la varianza y mejorar la distribución de los datos cuando estos tienen una distribución asimétrica.")
                else:
                 st.write("Por favor selecciona una columna para generar el histograma.")
    
    
    # Gráficos de densidad
    if section == "Gráficos de densidad":
        # Contenido relacionado con los gráficos de densidad
        titulo = "<h2 style='color:CornflowerBlue;'>Gráficos de densidad:</h2>"
        st.write(titulo, unsafe_allow_html=True)
        st.write("Estos dos graficos de densidad que proporcionan una descripción suave y continua de la distribución de datos, lo que permite identificar patrones de concentración y dispersión de manera más precisa que un histograma.")
        st.write("### Precio")

        col1, col2 = st.columns(2)
        with col1:

            # Create distplot with custom bin_size
            fig = px.histogram(df, x="Precio")
            st.plotly_chart(fig)
            
        with col2:
            fig = px.box(df, x="Precio")
            st.plotly_chart(fig)

        st.write("### Precio (Log)")
        st.write("El precio logarítmico es una transformación aplicada al precio original utilizando el logaritmo natural. "
                        "Esta transformación se utiliza para estabilizar la varianza y mejorar la distribución de los datos cuando estos tienen una distribución asimétrica.")
        col1, col2 = st.columns(2)
        with col1:

            # Create distplot with custom bin_size
            fig = px.histogram(df, x="Precio_log")
            st.plotly_chart(fig)
            
        with col2:
            fig = px.box(df, x="Precio_log")
            st.plotly_chart(fig)


    # Gráficos de Plotly
    if section == "Boxplot":
        titulo = "<h2 style='color:CornflowerBlue;'>Boxplot:</h2>"
        st.write(titulo, unsafe_allow_html=True)
        st.write("El Boxplot es una forma de visualizar la distribución y la variabilidad de un conjunto de datos. Consiste en una caja que muestra la mediana y los cuartiles del conjunto de datos, con -bigotes- que se extienden hacia los valores extremos. Es útil para identificar la dispersión de los datos, los valores atípicos y comparar distribuciones entre diferentes grupos.")         
        st.write("En el primer boxplot vemos el precio logaritmico (que mejora la distribucion y visualización de los datos) con los diferentes sistemas de combustión. En el segundo gráfico, lo relacionamos con la carrocería")
        
        # Boxplot 1
        st.write("### Boxplot 1 - Combustible & Precio logarítmico")
        fig = px.box(data_frame=df, x="Combustible", y="Precio_log", points="suspectedoutliers", hover_name="Marca")
        fig.update_layout(autosize=False, width=1200, height=600)
        st.plotly_chart(fig)
        st.markdown("---")

        # Boxplot 2
        st.write("### Boxplot 2 - Carrocería & Precio logarítmico")
        st.markdown("---")
        fig = px.box(data_frame=df, x="Tipo de carrocería", y="Precio_log", points="suspectedoutliers", hover_name="Marca")
        fig.update_layout(autosize=False, width=1200, height=600)
        st.plotly_chart(fig)
    
        # Boxplot 3
        st.write("### Boxplot 3 - Etiqueta medioambiental & Precio logarítmico")
        st.markdown("---")
        fig = px.box(data_frame=df, x="Etiqueta medioambiental", y="Precio_log", points="suspectedoutliers", hover_name="Marca")
        fig.update_layout(autosize=False, width=1200, height=600)
        st.plotly_chart(fig)

    if section == "Sunburst: Relación de carroceria con etiqueta medioambiental & precio":    
        titulo = "<h2 style='color:CornflowerBlue;'>Relación de carroceria con etiqueta medioambiental & precio:</h2>"
        col1, col2 = st.columns(2)
        with col1:
            st.write(titulo, unsafe_allow_html=True)    
            st.write("Al rededor del 70% de coches que se venden de segunda mano son de etiqueta medioambiental C.\n\n"
                        "Confirmando lo visto en el histograma, notamos también que los choches de etiqueta C son más económicos que los ECO y Cero. Sin embargo, se aprecia que aquellos con etiqueta B son los más económicos del mercado\n\n" 
                        "Esto posiblemente se deba a que solo el 23% de los vehículos de categoria B son todoterreno, contra 40% de los de etiqueta C, 50% de etiqueta Cero y 65% de etiqueta ECO.\n\n"
                        "Los coches de carrocería turismo también tienen una alta correlación con la etiqueta medioambiental y el precio.\n\n"
                        "Este gráfico nos permite confluir que el tipo de carrocería de un coche es determinante tanto para su precio, y que su distribución de cuenta también de la carrocería de los coches.")

        with col2:

            fig = px.sunburst(data_frame=df, path=["Etiqueta medioambiental","Tipo de carrocería"], hover_name="Tipo de carrocería", color="Precio")
            fig.update_traces(textinfo="label+percent parent")
            fig.update_layout(autosize=False, width=800, height=600)
            st.plotly_chart(fig)


    if section == "Barplot: Precio Promedio":
        titulo = "<h2 style='color:CornflowerBlue;'>Precio promedio por marca:</h2>"
        st.write(titulo, unsafe_allow_html=True)
        promedio_precio_por_marca = df.groupby("Marca")["Precio"].mean().reset_index()
        
        # Ordenar por precio promedio
        promedio_precio_por_marca = promedio_precio_por_marca.sort_values(by="Precio", ascending=False)
        
        # Obtener el top 5 de marcas con el precio promedio más alto
        top5_mas_altos = promedio_precio_por_marca.head(5)
        
        # Obtener el top 5 de marcas con el precio promedio más bajo
        top5_mas_bajos = promedio_precio_por_marca.tail(5)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Crear y mostrar el gráfico para el top 5 de marcas con precio promedio más alto
            fig_mas_altos = px.bar(top5_mas_altos, x="Marca", y="Precio", title="Top 5 de Marcas con Precio Promedio Más Alto")
            st.plotly_chart(fig_mas_altos)
        
        with col2:
            # Crear y mostrar el gráfico para el top 5 de marcas con precio promedio más bajo
            fig_mas_bajos = px.bar(top5_mas_bajos, x="Marca", y="Precio", title="Top 5 de Marcas con Precio Promedio Más Bajo")
            st.plotly_chart(fig_mas_bajos)

        titulo = "<h2 style='color:CornflowerBlue;'>Precio promedio por ciudad:</h2>"
        st.write(titulo, unsafe_allow_html=True)
        promedio_precio_por_ciudad = df.groupby("Ciudad")["Precio"].mean().reset_index()
    
        # Ordenar por precio promedio
        promedio_precio_por_ciudad = promedio_precio_por_ciudad.sort_values(by="Precio", ascending=False)
    
        # Obtener el top 5 de ciudades con el precio promedio más alto
        top5_mas_altos_ciudad = promedio_precio_por_ciudad.head(5)
    
        # Obtener el top 5 de ciudades con el precio promedio más bajo
        top5_mas_bajos_ciudad = promedio_precio_por_ciudad.tail(5)
    
        col1, col2 = st.columns(2)

        with col1:

            # Crear y mostrar el gráfico para el top 5 de ciudades con precio promedio más alto
            st.write("Vemos que en Daimiel (Ciudad Real) es donde mas caro se venden los coches usados seguido de cerca por Perillo (municipio de Oleiros, La Coruña)")
            fig_mas_altos_ciudad = px.bar(top5_mas_altos_ciudad, x="Ciudad", y="Precio", title="Top 5 de Ciudades con Precio Promedio Más Alto")
            st.plotly_chart(fig_mas_altos_ciudad)

        with col2:
            # Crear y mostrar el gráfico para el top 5 de ciudades con precio promedio más bajo
            st.write("Los precios mas bajos se encuentran Burjassot es un municipio ubicado en la provincia de Valencia y Bormujos es un municipio español situado en la provincia de Sevilla,")
            fig_mas_bajos_ciudad = px.bar(top5_mas_bajos_ciudad, x="Ciudad", y="Precio", title="Top 5 de Ciudades con Precio Promedio Más Bajo")
            st.plotly_chart(fig_mas_bajos_ciudad)
    
    #color
    if section == "¿Influye el color en el precio?":
            titulo = "<h2 style='color:CornflowerBlue;'>¿Influye el color en el precio?</h2>"
            st.write(titulo, unsafe_allow_html=True)
            # Calcular el precio promedio por color
            promedio_precio_por_color = df.groupby("Color")["Precio"].mean().reset_index()
            promedio_precio_por_color = promedio_precio_por_color[promedio_precio_por_color["Color"] != "Otro"]
            st.write("Observamos que dos colores destacan con un precio más elevado, siendo estos el negro, y, sorprendentemente, el verde. Los colores más económicos son el violeta y el beige. En el resto de los colores, se observa mayor uniformidad respecto al precio. Salvo estos 4 colores mencionados, el color no es un factor determinante en el precio.")
            st.markdown("---")
            
            # Crear el bar plot
            
            # Creo diccionario con los codigos RGB de los colores que están en x
            color_dict = {
                'Amarillo': 'rgb(255, 255, 102)',
                'Azul': 'rgb(0, 102, 204)',
                'Beige': 'rgb(245, 245, 220)',
                'Blanco': 'rgb(255, 255, 255)',
                'Gris': 'rgb(128, 128, 128)',
                'Marron': 'rgb(102, 51, 0)',
                'Naranja': 'rgb(255, 165, 0)',
                'Negro': 'rgb(0, 0, 0)',
                'Rojo': 'rgb(255, 255, 0)',
                'Verde': 'rgb(0, 204, 102)',
                'Violeta': 'rgb(178, 102, 255)'
            }
        
            colors = promedio_precio_por_color['Color'].map(color_dict).tolist()

            fig = go.Figure(data=[go.Bar(
                x=promedio_precio_por_color["Color"], 
                y=promedio_precio_por_color["Precio"], 
                marker_color=colors, 
                marker_line_color='rgb(0,0,0)', 
                marker_line_width=1.5, 
            )])

            fig.update_layout(title_text='Precio Promedio por Color', autosize=False, width=1000, height=400)
            st.plotly_chart(fig, align="left")
        
    #heatmap
    if section == "Heatmap: Variables con Mayor Correlación":
        titulo = "<h2 style='color:CornflowerBlue;'>Heatmap: Variables con Mayor Correlación</h2>"
        st.write(titulo, unsafe_allow_html=True)
        st.write("Aquí observamos las variables con un valor de correlación absoluto mayor a 0.3 respecto al precio.\n\n"
                "Aquellas variables que muestran correlación positiva, tienden a acompañar las fluctuaciones en el precio en el mismo sentido que este, mientras que las que tienen correlación negativa se mueven de manera inversa a los cambios en el precio.\n\n"
                "Podemos observar que tanto el kilometraje, los años y la cantidad de puertas tienen una correlación negativa al precio. El kilometraje se explica solo, mientras más ha sido utilizado el coche menor es su valor (con excepción en ocasiones de coches clásicos). El número de puertas probablemente esté asociado a que los autos con mayor cantidad de puertas son los de carrocerias familiares y de transporte, los cuales ya vimos en el sunburst tienen menor valor. Terminando con las correlaciones negativas, las emisiones son un reflejo de la etiqueta medioambiental, la cual también apreciamos en el sunburst que acompañaba al precio y tipo de carrocería.\n\n"
                "Finalmente, el alto da cuenta de las dimensiones del coche, y recordándo nuevamente al sunburst, observamos que los coches todoterreno son los de mayor precio, los cuales suelen ser más grandes que otras carrocerias (con excepción a las de transporte, pero estas son muy minoritarías, ergo, no influyen significativamente en la gráfica) ")
            
        # Dropeo variables no numéricas y las que tiene transformaciones logarítmicas
        df = df.select_dtypes(include=[np.number]) 
        df = df.drop(["KM_log","Precio_log"], axis=1)       

        #heatmap
        corr = df.corr()
        corr_precio = corr.loc[:, ['Precio']]
        corr_precio = corr_precio[(corr_precio['Precio'] > 0.3) | (corr_precio['Precio'] < -0.3) !=1]
        corr_precio = corr_precio.sort_values(by='Precio', ascending=False)

        plt.figure(figsize = (12, 8))
        sns.heatmap(corr_precio, annot = True)
        st.pyplot(plt) 
        


    #feature importance
    if section == "Feature Importance":
        titulo = "<h2 style='color:CornflowerBlue;'>Feature Importance</h2>"
        st.write(titulo, unsafe_allow_html=True)
        st.write("Utilizamos el modelo de Random Forest para la predicción de las variables con mayor relevancia para la predicción de precios.\n\n ")
        
        # Preparo los datos apra el modelo
        df = df.drop(["Precio_log"], axis=1)
        df[df.select_dtypes(exclude=['int', 'float']).columns] = df.select_dtypes(exclude=['int', 'float']).apply(lambda x: LabelEncoder().fit_transform(x))
                
        X = df.drop(["Precio"], axis = 1)
        y = df["Precio"]

        print(f"X: {X.shape}")
        print(f"y: {y.shape}")

        # Modelo de RandomForest para obtener Feature Importance
        model = pickle.load(open(f"../modelos_entrenados/ModeloRandomForest1.pk", 'rb'))
        # Calculamos Feature Importance
        importances = model.feature_importances_

        df_importances = pd.DataFrame(data = zip(X.columns, importances),
                                    columns = ["Columnas", "Importancia"])

        df_importances = df_importances.sort_values("Importancia", ascending = True)

        fig = go.Figure(go.Bar(
            x= df_importances["Importancia"],
            y=df_importances["Columnas"],
            orientation='h'))
        fig.update_layout(autosize=False, width=800, height=800)
        st.plotly_chart(fig)
    
    if section == "Conclusión":
        titulo = "<h2 style='color:CornflowerBlue;'>Conclusión:</h2>"
        st.write(titulo, unsafe_allow_html=True)
        st.write("A raiz de nuestras observaciones podemos inferir que el kilometraje, año, dimensiones y potencia son variables que influyen considerablemente en el precio.\n\n" 
                 "También hemos visto que varias de ellas tienen correlación entre si, como por ejemplo la etiqueta medioambiental y la carrocería. Y si bien estas tienen una correlación con el precio, no tiene ni cercanamente el mismo peso en el Feature Importance que las variables mencionadas arriba.\n\n"
                 "Finalmente, así como hemos podido ver cuales son las variables que tienen mayor incidencia para la predicción del precio, hemos también podido apreciar que el color, el combustible, la etiqueta medioambiental, el tipo de carroceria, el número de puertas y la garantía tienen muy bajo peso a la hora de determinar el precio.\n\n"
                 "Con esta información en mente, probaremos diferentes modelos para determinar cual ofrece mejores predicciones.")

if __name__ == "__eda_app__":
    eda_app()