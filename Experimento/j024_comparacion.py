import pandas as pd
import os
import matplotlib.pyplot as plt

# Rutas y parámetros
input_dir = "/home/mili_irusta/buckets/b1/exp/KA4210/"  # Directorio con los cortes generados
dataset_competencia = "/home/mili_irusta/buckets/b1/datasets/competencia_02.csv.gz"  # Archivo de datos reales
experimento_dir = "/home/mili_irusta/buckets/b1/datasets/dmeyf2024/Experimento/"
training_month = 202106  # Mes objetivo
cortes = range(9000, 13001, 500)  # Cortes a evaluar
semillas = [173717, 481129, 839203, 761081, 594017]  # Semillas utilizadas

# Cargar y filtrar los datos reales
print("Cargando y procesando datos reales...")
datos_reales = pd.read_csv(dataset_competencia, compression="gzip")
datos_reales = datos_reales[datos_reales["foto_mes"] == training_month]
datos_reales["clase_real"] = datos_reales["clase_ternaria"].apply(lambda x: 1 if x in ["BAJA+1", "BAJA+2"] else 0)
datos_reales = datos_reales[["numero_de_cliente", "clase_real"]]
print("Distribución de clase_real:")
print(datos_reales["clase_real"].value_counts())

# Función para calcular la ganancia con detalles de TP y FP
def calcular_ganancia_detallada(data):
    TP = ((data["Predicted"] == 1) & (data["clase_real"] == 1)).sum()
    FP = ((data["Predicted"] == 1) & (data["clase_real"] == 0)).sum()
    return TP * 7800 - FP * 2000, TP, FP

# Inicializar estructuras para resultados promediados
ganancias_original_promedio = {corte: [] for corte in cortes}
ganancias_estacional_promedio = {corte: [] for corte in cortes}

# Listar archivos en el directorio para verificar
print("Archivos disponibles en el directorio:")
disponibles = os.listdir(input_dir)
print(disponibles)

# Procesar cada corte para cada semilla
for semilla in semillas:
    print(f"Procesando semilla: {semilla}")
    for corte in cortes:
        print(f"Procesando corte: {corte}")

        # Construir nombres de archivos
        file_original = os.path.join(input_dir, f"KA4210_{corte}_original_seed_{semilla}.csv")
        file_estacional = os.path.join(input_dir, f"KA4210_{corte}_estacional_seed_{semilla}.csv")

        # Verificar existencia de archivos
        if not os.path.exists(file_original):
            print(f"Archivo no encontrado: {file_original}")
            continue
        if not os.path.exists(file_estacional):
            print(f"Archivo no encontrado: {file_estacional}")
            continue

        # Cargar predicciones
        predicciones_original = pd.read_csv(file_original)
        predicciones_estacional = pd.read_csv(file_estacional)

        # Unir predicciones con datos reales
        pred_original = predicciones_original.merge(datos_reales, on="numero_de_cliente")
        pred_estacional = predicciones_estacional.merge(datos_reales, on="numero_de_cliente")

        # Calcular ganancias con detalles de TP y FP
        ganancia_original, tp_original, fp_original = calcular_ganancia_detallada(pred_original)
        ganancia_estacional, tp_estacional, fp_estacional = calcular_ganancia_detallada(pred_estacional)

        # Agregar ganancias a la lista por semilla
        ganancias_original_promedio[corte].append(ganancia_original)
        ganancias_estacional_promedio[corte].append(ganancia_estacional)

# Calcular promedios por corte, manejando casos donde no hay datos
ganancias_original = [
    sum(ganancias_original_promedio[corte]) / len(ganancias_original_promedio[corte])
    if ganancias_original_promedio[corte] else 0
    for corte in cortes
]
ganancias_estacional = [
    sum(ganancias_estacional_promedio[corte]) / len(ganancias_estacional_promedio[corte])
    if ganancias_estacional_promedio[corte] else 0
    for corte in cortes
]

# Crear gráfico comparativo de ganancias promediadas
plt.figure(figsize=(10, 6))
plt.plot(cortes, ganancias_original, label="Original (Promedio)", marker="o", linestyle="--", color="blue")
plt.plot(cortes, ganancias_estacional, label="Estacional (Promedio)", marker="o", linestyle="--", color="green")
plt.xlabel("Corte")
plt.ylabel("Ganancia Promedio")
plt.title("Comparación de Ganancias Promedio por Modelo y Corte")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Guardar el gráfico
output_graph = os.path.join(experimento_dir, "comparacion_ganancias_promedio.png")
plt.savefig(output_graph)
plt.show()

print(f"Gráfico guardado en: {output_graph}")
