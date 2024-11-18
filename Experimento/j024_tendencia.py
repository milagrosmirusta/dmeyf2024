#tendencia_dataset.py


import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Ruta del archivo
file_path = "~/buckets/b1/datasets/competencia_02.csv.gz"
output_pdf = "~/buckets/b1/datasets/dmeyf2024/Experimento/tendencia.pdf"

print("Cargando el dataset, esto puede llevar algo de tiempo...")
# Cargar el dataset con low_memory=False para evitar el DtypeWarning
dataset = pd.read_csv(file_path, compression='gzip', low_memory=False)

# Convertir 'foto_mes' en un formato de fecha, asumiendo que representa el año y el mes (e.g., 202301 para enero de 2023)
dataset['foto_mes'] = pd.to_datetime(dataset['foto_mes'], format='%Y%m', errors='coerce')

# Eliminar filas con valores nulos en 'foto_mes' tras la conversión
dataset = dataset.dropna(subset=['foto_mes'])

# Excluir 'numero_de_cliente' si está presente
if 'numero_de_cliente' in dataset.columns:
    dataset = dataset.drop(columns=['numero_de_cliente'])

# Filtrar solo columnas numéricas y 'foto_mes'
dataset = dataset[['foto_mes'] + dataset.select_dtypes(include='number').columns.tolist()]

print("Calculando la media por cada atributo y cada foto_mes...")
# Agrupar por 'foto_mes' y calcular la media de las columnas numéricas para cada mes
medias_por_mes = dataset.groupby('foto_mes').mean().reset_index()

print(medias_por_mes.head())  # Mostrar los primeros resultados para verificación

print("Generando gráficos de tendencia para cada atributo...")
# Crear directorio si no existe
os.makedirs(os.path.dirname(output_pdf), exist_ok=True)

# Crear un archivo PDF para guardar los gráficos
with PdfPages(output_pdf) as pdf:
    for columna in medias_por_mes.columns[1:]:
        plt.figure(figsize=(10, 6))
        plt.plot(medias_por_mes['foto_mes'], medias_por_mes[columna], marker='o', color='blue')
        plt.title(f"Tendencia del atributo: {columna}")
        plt.xlabel("Fecha (Mes)")
        plt.ylabel("Media del atributo")
        plt.xticks(rotation=45)
        plt.grid(visible=True)
        
        # Guardar el gráfico en el PDF
        pdf.savefig()
        plt.close()

print(f"Gráficos de tendencia guardados en: {output_pdf}")

