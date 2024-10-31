#pipeline_feature_estacionalidad.py

# Importar las funciones de los scripts existentes
from cargar_datasets import cargar_datasets
from filtrar_2020 import filtrar_2020
import numpy as np
import polars as pl
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split


def entrenar_lightgbm(df, target_column):
    """
    Entrena un modelo de LightGBM y obtiene las tres características más importantes.

    Args:
        df (pd.DataFrame): DataFrame con los datos de 2020.
        target_column (str): Nombre de la columna objetivo.

    Returns:
        list: Lista de las tres características más importantes.
    """
    # Asegurarse de que todas las columnas son numéricas
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # Reemplazar NaN con un valor, por ejemplo 0, si es necesario
    df.fillna(0, inplace=True)

    # Definir las características (X) y la variable objetivo (y)
    X = df.drop(columns=['clase_ternaria','foto_mes','numero_de_cliente'])
    y = df[target_column].map(lambda x: 0 if x == "CONTINUA" else 1)

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear el dataset de LightGBM
    lgb_train = lgb.Dataset(X_train, y_train)

    # Configuración del modelo
    params = {
        'objective': 'binary',
        'learning_rate': 0.01,
        'max_bin': 255,
        'min_data_in_leaf': 50,
        'verbose': -1,
    }

    # Entrenar el modelo
    gbm = lgb.train(params, lgb_train, num_boost_round=100)

    # Obtener la importancia de las características
    feature_importance = pd.DataFrame({
        'Feature': gbm.feature_name(),
        'Importance': gbm.feature_importance()
    })

    # Ordenar por importancia y obtener la más importante
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
    top_features = feature_importance['Feature'].head(3).tolist()
    print("Top 3 features:", top_features)
    return top_features
    
def add_seasonal_components(df, date_col):
    df = df.with_columns([
        (pl.col(date_col) % 100).alias("month")
    ])
    df = df.with_columns([
        (2 * np.pi * pl.col("month") / 12).sin().alias("sin_month"),
        (2 * np.pi * pl.col("month") / 12).cos().alias("cos_month")
    ])
    return df

def crear_lags(df, top_features):
    """
    Crea lags de 1 y 2 meses para las características más importantes seleccionadas.

    Args:
        df (pd.DataFrame): DataFrame original.
        top_features (list): Lista de las características más importantes.

    Returns:
        pd.DataFrame: DataFrame con lags añadidos.
    """
    # Asegúrate de que el DataFrame esté ordenado por `numero_de_cliente` y `foto_mes`
    df = df.sort_values(by=['numero_de_cliente', 'foto_mes'])

    # Crear lags para cada una de las top features
    for var in top_features:
        df[f'{var}_lag1'] = df.groupby('numero_de_cliente')[var].shift(1)
        df[f'{var}_lag2'] = df.groupby('numero_de_cliente')[var].shift(2)

    return df

# Ejecución del pipeline
if __name__ == "__main__":
    # Cargar los datasets
    df_polars = cargar_datasets()

    # Filtrar los datos de 2020 usando la función de `filtrar_2020.py`
    df_2020 = filtrar_2020(df_polars)

    # Añadir componentes estacionales directamente en el DataFrame de Polars
    df_with_seasonality_2020 = add_seasonal_components(df_2020, "foto_mes")

    # Convertir a Pandas para usar con LightGBM
    df_with_seasonality_2020 = df_with_seasonality_2020.to_pandas()

    # Entrenar el modelo y obtener las tres características más importantes
    top_features = entrenar_lightgbm(df_with_seasonality_2020, target_column="clase_ternaria")

    # Crear lags de las características más importantes
    df_final = crear_lags(df_with_seasonality_2020, top_features)

    # Mostrar los resultados
    print("Pipeline completo. DataFrame final:")
    print(df_final.head(10))