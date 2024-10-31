from cargar_datasets import cargar_datasets
import polars as pl
import pandas as pd 

# Definir la función para asegurar que el DataFrame esté en formato Polars
def asegurar_polars(df):
    if isinstance(df, pd.DataFrame):
        print("Convirtiendo de Pandas a Polars.")
        return pl.from_pandas(df)
    elif isinstance(df, pl.DataFrame):
        print("Ya es un DataFrame de Polars.")
        return df
    else:
        raise TypeError("El tipo del DataFrame no es ni Pandas ni Polars.")
        
def filtrar_2020(df):
    # Asegurarse de que el DataFrame esté en formato Polars
    df_polars = asegurar_polars(df)
    
    # Filtrar los datos de 2020
    df_2020 = df_polars.filter(
        (pl.col("foto_mes") >= 202001) & (pl.col("foto_mes") <= 202012)
    )

    return df_2020 

# Ejecutar el filtrado si el script se corre directamente
if __name__ == "__main__":
    # Cargar los datasets usando cargar_datasets.py
    df_polars = cargar_datasets()

    # Filtrar los datos de 2020
    df_2020  = filtrar_2020(df_polars)

    # Mostrar los resultados
    print("Datos de 2020:")
    print(df_2020)

