# comparar_lightgbm.py
# Importar las funciones y librerías necesarias
from cargar_datasets import cargar_datasets
from filtrar_2020 import filtrar_2020, asegurar_polars
from pipeline_feature_estacionalidad import entrenar_lightgbm, add_seasonal_components, crear_lags

import lightgbm as lgb
import pandas as pd
import polars as pl
from sklearn.metrics import accuracy_score, roc_auc_score



def preparar_junio_2021(df):
    """
    Filtra los datos de junio de 2021 usando Polars y luego convierte a Pandas.

    Args:
        df (pl.DataFrame): DataFrame original de Polars.

    Returns:
        pd.DataFrame: DataFrame con datos de junio de 2021 en formato Pandas.
    """
    # Asegurar que estamos trabajando con un DataFrame de Polars
    df = asegurar_polars(df)
    
    # Filtrar los datos de junio de 2021
    df_june_2021 = df.filter(pl.col("foto_mes") == 202106)
    
    # Convertir a Pandas para su uso posterior
    return df_june_2021.to_pandas()
    
def entrenar_y_predecir(df_train, df_test, target_column="clase_ternaria"):
    """
    Entrena un modelo de LightGBM y realiza predicciones sobre un conjunto de prueba.

    Args:
        df_train (pd.DataFrame): DataFrame de entrenamiento.
        df_test (pd.DataFrame): DataFrame de prueba.
        target_column (str): Nombre de la columna objetivo.

    Returns:
        tuple: Predicciones y métricas de evaluación (precisión y AUC).
    """
    # Preparar datos de entrenamiento
    df_train = df_train.apply(pd.to_numeric, errors='coerce')
    df_test = df_test.apply(pd.to_numeric, errors='coerce')
    
    X_train = df_train.drop(columns=[target_column])
    y_train = df_train[target_column].map(lambda x: 0 if x == "CONTINUA" else 1)
    
    # Preparar datos de prueba
    X_test = df_test.drop(columns=[target_column])
    y_test = df_test[target_column].map(lambda x: 0 if x == "CONTINUA" else 1)

     # Verificar que ambos conjuntos contengan al menos una muestra de cada clase
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        print("Advertencia: Uno de los conjuntos contiene solo una clase. AUC no se puede calcular.")
        return None, {"accuracy": None, "auc": None}
    
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

    # Predecir en los datos de prueba
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    y_pred_binary = [1 if pred > 0.5 else 0 for pred in y_pred]

      # Cálculo de AUC con manejo de errores
    try:
        auc_score = roc_auc_score(y_test, y_pred)
    except ValueError as e:
        warnings.warn("No se puede calcular AUC debido a clases únicas en el conjunto de prueba.", UndefinedMetricWarning)
        auc_score = None

    return y_pred_binary, {"accuracy": accuracy, "auc": auc_score}

# Ejecución del pipeline de comparación
if __name__ == "__main__":
    # Cargar los datasets y asegurarse de que están en Polars
    df_polars = asegurar_polars(cargar_datasets())
    
    # Filtrar datos de 2020 y junio de 2021
    df_2020 = filtrar_2020(df_polars).to_pandas()
    df_june_2021 = preparar_junio_2021(df_polars)
    
    # Evaluación sin estacionalidad ni lags
    print("Evaluando modelo sin estacionalidad ni lags para predecir junio 2021...")
    pred_sin_estacionalidad, resultados_sin_estacionalidad = entrenar_y_predecir(df_2020, df_june_2021)
    print("Resultados sin estacionalidad:", resultados_sin_estacionalidad)

    # Preparar dataset con estacionalidad y lags
    df_with_seasonality_2020 = add_seasonal_components(filtrar_2020(df_polars), "foto_mes").to_pandas()

    # Obtener las tres características más importantes del modelo con estacionalidad
    top_features = entrenar_lightgbm(df_with_seasonality_2020, target_column="clase_ternaria")

    # Crear lags de las características más importantes
    df_final = crear_lags(df_with_seasonality_2020, top_features)

    # Evaluación con estacionalidad y lags
    print("Evaluando modelo con estacionalidad y lags para predecir junio 2021...")
    pred_con_estacionalidad, resultados_con_estacionalidad = entrenar_y_predecir(df_final, df_june_2021)
    print("Resultados con estacionalidad:", resultados_con_estacionalidad)

    # Comparar los resultados
    print("\nComparación de Modelos para Junio 2021:")
    print("Sin estacionalidad - Precisión:", resultados_sin_estacionalidad["accuracy"], "| AUC:", resultados_sin_estacionalidad["auc"])
    print("Con estacionalidad - Precisión:", resultados_con_estacionalidad["accuracy"], "| AUC:", resultados_con_estacionalidad["auc"])