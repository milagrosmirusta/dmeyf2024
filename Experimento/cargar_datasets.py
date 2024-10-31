# cargar_datasets.py

# Importar las librerías necesarias
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import polars as pl


def cargar_datasets():
    """
    Función para cargar los datasets con y sin estacionalidad.
    
    Returns:
        df_with_seasonality (pd.DataFrame): DataFrame con estacionalidad y lags.
        df_without_seasonality (pd.DataFrame): DataFrame sin estacionalidad ni lags.
    """
    # Cargar el dataset con estacionalidad
    df = pl.read_csv('/home/mili_irusta/buckets/b1/datasets/competencia_02.csv.gz', 
    infer_schema_length=10000,
    ignore_errors=True)
    df = df.to_pandas()


    return df
# Ejecutar la carga de datasets si el script se corre directamente
if __name__ == "__main__":
    df = cargar_datasets()
    print("Dataset cargado exitosamente")