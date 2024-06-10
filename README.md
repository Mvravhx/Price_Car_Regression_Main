### Paso 1:
Descargar o clonar el proyecto => `git clone enlace_repositorio`<br>
Acceder a la carpeta del proyecto => `cd Price_car_regresion`<br>

### Paso 2:
Crear un entorno virtual => `python -m venv nombre_entorno`

### Paso 3:
Activar el entorno virtual <br>
En windows => `source nombre_entorno/Scripts/activate` en caso de que devuelva error probar con=> `.\nombre_entorno\Scripts\activate`<br>
En linux => `source nombre_entorno/bin/activate`

### Paso 4:
Instalar todas las dependencias necesarias para el proyecto a través de => `pip install -r requirements.txt`

### Paso 5:
Crear una carpeta llamada modelos_entrenados en el que guardar los modelos => `mkdir modelos_entrenados` <br> 
Ejecutar el script RandomForest.py para entrenar el modelo que se usará en Streamlit => <br>
`python ./modelos/RandomForest.py` <br> 

### Paso 6:
Acceder a la carpeta del proyecto de streamlit: `cd ./machine_learning/`

### Paso 7:
Iniciar el servidor de streamlit:  `streamlit run "machine_learning\app.py"`

> [!WARNING]  
> En el paso 7 la ruta especificada es la ruta relativa, de no funcionar es necesario especificar la ruta completa al archivo.
> En visual estudio lo puedes hacer con click derecho sobre el archivo y copiar ruta.

> [!NOTE]  
> Se recomienda utilizar una consola gitbash para el correcto funcionamiendo a la hora de configurar el entorno virtual de python.
> 
