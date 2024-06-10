import streamlit as st
import pandas as pd
import pickle
import numpy as np
import math
import plotly.express as px

def estimar_app():
    st.title("Introduce los datos de tu coche")
    df = pd.read_csv('../data/datos_limpios.csv')

    datos_coche = {}

    datos_coche["Año"] = st.slider( label     = "Año modelo",
                            min_value = int(1990),
                            max_value = int(2024), 
                            step      = 1)
    
    marcas = df["Marca"].unique().tolist()

    datos_coche["marca"] = st.selectbox("Marca", options = marcas)

    # Input Numbers
    datos_coche["km"] = st.number_input(label = "Kilometros vehiculo",
                         min_value = 0,
                         max_value = 1000000,
                         value = 0,
                         step = 10)

    datos_coche["combustible"] = st.radio(label      = "Tipo de combustible:",
                            options    = ["Gasolina","Diésel","Híbrido","Eléctrico","Otros"],
                            index      = 0,
                            disabled   = False,
                            horizontal = True)
    
    datos_coche["cambio"] = st.radio(label      = "Tipo de cambio:",
                            options    = ["Automático","Manual","Otro"],
                            index      = 0,
                            disabled   = False,
                            horizontal = True)

    datos_coche["n_puertas"] = st.slider(label= "Nº de puertas",
                                min_value = int(2),
                                max_value = int(5), 
                                step      = 1)


    carrocerias = df["Tipo de carrocería"].unique().tolist()
    carrocerias.remove("Comercial grande")

    datos_coche["carroceria"] = st.selectbox("Tipo de carrocería", options = carrocerias)

    colores = df["Color"].unique().tolist()

    datos_coche["color"] = st.selectbox("Color", options = colores)

    datos_coche["potencia"] = st.slider(label     = "Potencia",
                                min_value = float(45),
                                max_value = float(df["Potencia"].max()), 
                                step      = 1.)
    
    datos_coche["velocidad"] = st.slider(label     = "Velocidad máxima",
                                min_value = int(110),
                                max_value = int(df["Velocidad máxima"].max()), 
                                step      = 1)
    
        
    datos_coche["aceleracion"] = st.slider(label     = "Aceleración",
                                min_value = float(2),
                                max_value = float(df["Aceleracion"].max()), 
                                step      = 0.1)
    
            
    datos_coche["consumo"] = st.slider(label     = "Consumo",
                                min_value = float(0.6),
                                max_value = float(df["Consumo"].max()), 
                                step      = 0.1)
    
    datos_coche["emisiones"] = st.slider(label     = "Emisiones",
                                min_value = int(0),
                                max_value = int(df["Emisiones"].max()), 
                                step      = 1)    

    etiquetas = df["Etiqueta medioambiental"].unique().tolist()
    etiquetas.remove("Cero/ECO")

    datos_coche["etiqueta"] = st.radio(label      = "Etiqueta medioambiental:",
                            options    = etiquetas,
                            index      = 0,
                            disabled   = False,
                            horizontal = True)
    st.write("Dimensiones")


    col1, col2, col3 = st.columns(3)

    with col1:
        datos_coche["largo"] = st.number_input(label = "Largo",
                     min_value = 0,
                     max_value = 1000000,
                     value = 0,
                     step = 10)
    with col2:
        datos_coche["ancho"] = st.number_input(label = "Ancho",
                     min_value = df["Ancho"].min(),
                     max_value = 1000000,
                     value = 0,
                     step = 10)
    with col3:
        datos_coche["alto"] = st.number_input(label = "Alto",
                     min_value = 0,
                     max_value = 1000000,
                     value = 0,
                     step = 10)

    if st.button(label = "Summit",
                 key   = "submit2",
                 type  = "primary"): # El type primary resalta el boton
        mostrar_estimado(datos_coche)




def mostrar_estimado(datos):

    tranformadores = {
        
        "KnnImp": pickle.load(open(f"../transformadores/KnnImputer_sin_precio.pk", 'rb')),
        "marca": pickle.load(open(f"../transformadores/marca_encoder.pk", 'rb')),
        "cambio": pickle.load(open(f"../transformadores/Cambio_encoder.pk", 'rb')),
        "color": pickle.load(open(f"../transformadores/Color_encoder.pk", 'rb')),
        "combustible": pickle.load(open(f"../transformadores/Combustible_encoder.pk", 'rb')),
        "etiqueta": pickle.load(open(f"../transformadores/Etiqueta_encoder.pk", 'rb')),
        "minmax": pickle.load(open(f"../transformadores/MinMax.pk", 'rb')),
        "carroceria": pickle.load(open(f"../transformadores/Tipo_carroceria_encoder.pk", 'rb')),
        
    }

    df_pro = pd.read_csv("../data/datos_procesados_2.csv")
    
    modelo = pickle.load(open(f"../modelos_entrenados/ModeloRandomForest1.pk", 'rb'))
    
    datos_procesados = np.array([])

    datos_procesados = np.concatenate((datos_procesados,np.array([datos["Año"]])))
    datos_procesados = np.concatenate((datos_procesados,np.array([datos["n_puertas"]])))
    datos_procesados = np.concatenate((datos_procesados,np.array([datos["potencia"]])))
    datos_procesados = np.concatenate((datos_procesados,np.array([datos["velocidad"]])))
    datos_procesados = np.concatenate((datos_procesados,np.array([datos["aceleracion"]])))
    datos_procesados = np.concatenate((datos_procesados,np.array([datos["consumo"]])))
    datos_procesados = np.concatenate((datos_procesados,np.array([datos["emisiones"]])))
    datos_procesados = np.concatenate((datos_procesados,np.array([datos["largo"]])))
    datos_procesados = np.concatenate((datos_procesados,np.array([datos["ancho"]])))
    datos_procesados = np.concatenate((datos_procesados,np.array([datos["alto"]])))
    datos_procesados = np.concatenate((datos_procesados,np.array([np.log(datos["km"] + 1)])))
    
            
    dato_transformado = tranformadores['combustible'].transform([[datos['combustible']]]).toarray().reshape(1,-1)[0]
    datos_procesados = np.concatenate((datos_procesados,dato_transformado))
    
    dato_transformado = tranformadores['cambio'].transform([[datos['cambio']]]).toarray().reshape(1,-1)[0]
    datos_procesados = np.concatenate((datos_procesados,dato_transformado))

    dato_transformado = tranformadores['color'].transform([[datos['color']]]).toarray().reshape(1,-1)[0]
    datos_procesados = np.concatenate((datos_procesados,dato_transformado))

    dato_transformado = tranformadores['marca'].transform([[datos['marca']]])
    datos_procesados = np.concatenate((datos_procesados,dato_transformado))

    dato_transformado = tranformadores['carroceria'].transform([[datos['carroceria']]]).toarray().reshape(1,-1)[0]
    datos_procesados = np.concatenate((datos_procesados,dato_transformado))

    dato_transformado = tranformadores['etiqueta'].transform([[datos['etiqueta']]]).toarray().reshape(1,-1)[0]
    datos_procesados = np.concatenate((datos_procesados,dato_transformado))

    df_procesado = pd.DataFrame(data =[datos_procesados], columns= df_pro.drop(['Precio_log'], axis=1).columns )

    datos_escalados = tranformadores['minmax'].transform(df_procesado)

    prediction = modelo.predict(datos_escalados)
    prediction = np.exp(prediction)+1

    st.write(f"#### Precio estimado para el model descrito: {prediction[0]}")

    # print(datos_escalados)
    print(df_pro.shape)
    
    st.write(f"##### Vehículos mas similares al introdicido:")
    st.write("En la siguiente gráfica se muestran los vehículos más parecidos, considerando los datos que se han introducido.Se muestra la marca y el precio para poder hacerse una idea más sólida del precio del coche descrito.")
    mostrar_grafica(datos_escalados,df_pro,modelo,tranformadores)


def mostrar_grafica(datos_vehiculo,df_procesado,modelo,tranformadores):
    def distancia_euclideana(P1, P2):
        return math.sqrt(sum([(y - x)**2 for x, y in zip(P1, P2)]))
    
    def distancia_manhattan(P1, P2):
    
        total = 0
        
        for x, y in zip(P1, P2):
            total += abs(y - x)
            
        return total
    
    df_escalado = tranformadores["minmax"].transform(df_procesado.drop(['Precio_log'], axis=1))

    distancias = []

    print(df_escalado[0].shape)
    for i in  range(df_escalado.shape[0]):
        coche = df_escalado[i]
        d_eu=distancia_euclideana(datos_vehiculo[0],coche)
        d_man=distancia_manhattan(datos_vehiculo[0],coche)

        distancias.append([d_eu,d_man,d_eu+d_man/2])

    df_distancia = pd.DataFrame(data=distancias, columns=["eucludanea","man","Total"])
    df_distancia["Marca"] = df_procesado["Marca_encoded"]
    df_distancia["Precio"] = df_procesado["Precio_log"]

    df_distancia["Marca"] = df_distancia["Marca"].apply(lambda x : tranformadores['marca'].inverse_transform(np.array([x]).astype(int)))
    df_distancia["Precio"] = df_distancia["Precio"].apply(lambda x : np.exp(x)+1 )


    df_distancia = df_distancia.sort_values("Total").head(100).reset_index()

    print(df_distancia.head(10))
    fig = px.scatter(df_distancia, x="eucludanea", y="man", hover_data=["Marca","Precio"])
    fig.update_layout(autosize=False, width=900, height=600)
    st.plotly_chart(fig)


if __name__ == "__estimar_app__":
    estimar_app()

