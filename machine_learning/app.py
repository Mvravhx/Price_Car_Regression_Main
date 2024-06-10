import streamlit as st
import plotly.express as px
import numpy as np
import os
from eda import eda_app
from ml import ml_app
from about import about_app
from estimar import estimar_app

def main():
    
    st.set_page_config(layout="wide")
    
    menu = ["Nuestro proyecto", "Exploratory Data Analysis", "Machine Learning Model", "Estimar precio"]

    choice = st.sidebar.selectbox(label = "Menu", options = menu, index = 0)

    c = st.container()

    if choice == "Nuestro proyecto":
        proyecto = "<h1 style='color:CornflowerBlue; text-decoration: underline;'>Predicción de precio de coches de segunda mano</h1>"
        titulo = "<h2 style='color:CornflowerBlue;'>Big Data & IA - dsb07rt - Grupo A - Proyecto Final</h2>"
        sub_titulo = "<h3 style='color:CornflowerBlue;'>Integrantes: Morales, Ángel - Navarro, Enrique - Ruiz Paniagua, Bárbara Lía - Zunzunegui, Fernando</h3>"
        c.write(proyecto, unsafe_allow_html=True)
        st.write(titulo, unsafe_allow_html=True)
        st.write(sub_titulo, unsafe_allow_html=True)
        st.markdown("---")

        st.write("A la hora de comprar o vender un coche de segunda mano, hay muchos factores que evaluar. Muchos de ellos son aspectos técnicos de cada vehículo en particular que hasta que haya que ver de forma física.\n\n"
                 "¡Hay tanto que considerar que uno puede marearse!\n\n" 
                 "Por eso, hemos decidido quitaros una parte de ese problema de vuestros hombros y brindaros esta herramienta para que podaís verificar cual es el precio estimado de mercado del coche de segunda mano en cuestión.\n\n "
                 "Con nuestro modelo de predicción de precios de coches de segunda mano no tendraís que preocuparte por saber cual es un precio justo de tu coche a la hora de venderlo. Y si lo que buscaís es comprar, podraís saber cual es el precio estimado de diferentes coches, considerando una serie de variables que te dejarán mucho más tranquilo a la hora de realizar una compra.\n\n"
                 )
        st.markdown("---")
        st.write("Para poder traeros esta información de la manera más confiable, hemos realizado un webscrapping de la página web [km77](https://www.km77.com), de la cual hemos recolectado los datos de más de 52.000 coches publicados, y hemos evaluado multiples variables para realizar entregaros los mejores resultados." 
        )
        st.markdown("---")
        st.markdown("En las próximas secciones podraís ver como hemos evaluado esta información, los modelos de predicción que hemos utilizado y, finalmente, predecir el precio del coche que desees.\n\n"
                    "¡Esperamos esta herramiente os sea de mucha ayuda y os ahorre un dolor de cabeza!")

    elif choice == "Exploratory Data Analysis":
        eda_app()

    elif choice == "Machine Learning Model":
        ml_app()

    elif choice  == "Estimar precio":
        estimar_app()
    else:
        about_app()




if __name__ == "__main__":
    main()