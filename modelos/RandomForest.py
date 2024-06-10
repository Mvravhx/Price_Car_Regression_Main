import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle
import plotly.express as px

from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')

df = pd.read_csv('./data/datos_procesados_2.csv')

X = df.drop(['Precio_log'], axis=1)
# Normalizo 
scaler_x = MinMaxScaler()
scaler_x.fit(X)

pickle.dump(scaler_x, open("./transformadores/MinMax.pk", 'wb'))


X = scaler_x.transform(X)

y = df['Precio_log']

X.shape, y.shape

df.drop(['Precio_log','Largo','Ancho','Alto'], axis=1).columns

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)

X_train.shape, y_train.shape, X_test.shape, y_test.shape

from sklearn.ensemble import  RandomForestRegressor

rf_1 = RandomForestRegressor()
rf_1.get_params()
print("Entrenando modelo definitivo")
rf_1.fit(X_train, y_train)

# Check training
y_pred_train = rf_1.predict(X_train)

pickle.dump(rf_1, open("./modelos_entrenados/ModeloRandomForest1.pk", 'wb'))

# rmse
from sklearn.metrics import  mean_squared_error

rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))

print('RMSE Train data {}'.format(rmse_train))


# error test data
y_pred_test = rf_1.predict(X_test)

rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

print('RMSE Test data {}'.format(rmse_test))