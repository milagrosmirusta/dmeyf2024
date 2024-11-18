import os
import pandas as pd
import numpy as np

# Ruta del dataset estacional
dataset_estacional = "~/buckets/b1/datasets/dmeyf2024/Experimento/dataset_con_variables_estacionales.csv.gz"

# Leer el dataset
print('Leyendo dataset')
dataset = pd.read_csv(dataset_estacional, compression='gzip')
print(f"Dataset cargado desde {dataset_estacional}. Dimensiones: {dataset.shape}")

# Ver los primeros valores únicos de la columna 'foto_mes'
print("Primeros valores de 'foto_mes':")
print(dataset['foto_mes'].head(10))

# Convertir 'foto_mes' al formato yyyymm
print("Convirtiendo 'foto_mes' al formato yyyymm...")
dataset['foto_mes'] = pd.to_datetime(dataset['foto_mes'], errors='coerce')  # Asegurarse de que sean fechas
dataset['foto_mes'] = dataset['foto_mes'].dt.strftime('%Y%m').astype(int)  # Convertir al formato yyyymm
print("Valores únicos en 'foto_mes' después de la conversión:")
print(dataset['foto_mes'].unique())

# Guardar el dataset procesado (opcional)
output_file = "~/buckets/b1/datasets/dmeyf2024/Experimento/dataset_con_variables_estacionales_limpio.csv.gz"
dataset.to_csv(output_file, index=False, compression='gzip')
print(f"Dataset procesado guardado en {output_file}.")
