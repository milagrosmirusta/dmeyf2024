# variables_estacionales.py

import os
import pandas as pd

# Ruta del archivo original y la salida
file_path = "~/buckets/b1/datasets/competencia_02.csv.gz"
output_file = "~/buckets/b1/datasets/dmeyf2024/Experimento/dataset_con_variables_estacionales.csv.gz"

print("Cargando el dataset, esto puede llevar algo de tiempo...")
# Cargar el dataset con low_memory=False para evitar el DtypeWarning
dataset = pd.read_csv(file_path, compression='gzip', low_memory=False)

# Convertir 'foto_mes' en un formato de fecha, asumiendo que representa el año y el mes (e.g., 202301 para enero de 2023)
dataset['foto_mes'] = pd.to_datetime(dataset['foto_mes'], format='%Y%m', errors='coerce')
dataset = dataset.dropna(subset=['foto_mes'])  # Eliminar filas con nulos en 'foto_mes'

# Filtrar solo columnas numéricas y 'foto_mes'
#dataset = dataset[['foto_mes'] + dataset.select_dtypes(include='number').columns.tolist()]

# 1. Variables de cambio estacional mes a mes
# Ejemplo: variación en saldo bancario y en cantidad de transacciones
dataset['diff_saldo'] = dataset['mcuentas_saldo'] - dataset['mcuentas_saldo'].shift(1)
dataset['ctransacciones'] = dataset['ctarjeta_debito_transacciones'] + dataset['ctarjeta_visa_transacciones'] + dataset['ctarjeta_master_transacciones']
dataset['diff_transacciones'] = dataset['ctransacciones'] - dataset['ctransacciones'].shift(1)

# 2. Promedios y variaciones en ventanas de tiempo
# Calcular el promedio móvil para el saldo y transaccion en ventanas de 3, 6 y 12 meses
dataset['saldo_3_meses'] = dataset['mcuentas_saldo'].rolling(window=3).mean()
dataset['saldo_6_meses'] = dataset['mcuentas_saldo'].rolling(window=6).mean()
dataset['saldo_12_meses'] = dataset['mcuentas_saldo'].rolling(window=12).mean()
dataset['transacciones_3_meses'] = dataset['ctransacciones'].rolling(window=3).mean()
dataset['transacciones_6_meses'] = dataset['ctransacciones'].rolling(window=6).mean()
dataset['transacciones_12_meses'] = dataset['ctransacciones'].rolling(window=12).mean()

# Cambio porcentual en ventanas de 6 meses para saldo
dataset['perc_change_saldo_6_meses'] = (dataset['mcuentas_saldo'] - dataset['mcuentas_saldo'].shift(6)) / dataset['mcuentas_saldo'].shift(6)
dataset['perc_change_trans_6_meses'] = (dataset['ctransacciones'] - dataset['ctransacciones'].shift(6)) / dataset['ctransacciones'].shift(6)

# 3. Variables de tendencia estacional
dataset['es_diciembre'] = dataset['foto_mes'].dt.month == 12
dataset['trimestre_4'] = dataset['foto_mes'].dt.quarter == 4

# Índice de variación estacional para saldo (comparando con la media anual)
annual_mean_saldo = dataset.groupby(dataset['foto_mes'].dt.year)['mcuentas_saldo'].transform('mean')
dataset['indice_saldo'] = dataset['mcuentas_saldo'] / annual_mean_saldo

# Índice de variación estacional para transacciones (comparando con la media anual)
annual_mean_tr = dataset.groupby(dataset['foto_mes'].dt.year)['ctransacciones'].transform('mean')
dataset['indice_transacciones'] = dataset['ctransacciones'] / annual_mean_tr

print("Cálculo de variables estacionales completado.")

# Crear directorio de salida si no existe
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Guardar el dataset con las nuevas variables
dataset.to_csv(output_file, index=False, compression='gzip')
print(f"Dataset con variables estacionales guardado en: {output_file}")
