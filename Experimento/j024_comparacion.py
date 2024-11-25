import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Función para formatear en miles
def format_miles(x, pos):
    return f'{x:,.0f}'  # Formato con separadores de miles

# Rutas y parámetros
input_dir = "/home/mili_irusta/buckets/b1/exp/KA4210/"
dataset_competencia = "/home/mili_irusta/buckets/b1/datasets/competencia_02.csv.gz"
experimento_dir = "/home/mili_irusta/buckets/b1/datasets/dmeyf2024/Experimento/"
training_month = 202106
cortes = range(9000, 13001, 500)
semillas = [173717, 481129, 839203, 761081, 594017]

# Cargar y filtrar los datos reales
print("Cargando y procesando datos reales...")
datos_reales = pd.read_csv(dataset_competencia, compression="gzip")
datos_reales = datos_reales[datos_reales["foto_mes"] == training_month]
datos_reales["clase_real"] = datos_reales["clase_ternaria"].apply(lambda x: 1 if x in ["BAJA+2"] else 0)
datos_reales = datos_reales[["numero_de_cliente", "clase_real"]]
print("Distribución de clase_real:")
print(datos_reales["clase_real"].value_counts())

# Función para calcular la ganancia
def calcular_ganancia_detallada(data):
    TP = ((data["Predicted"] == 1) & (data["clase_real"] == 1)).sum()
    FP = ((data["Predicted"] == 1) & (data["clase_real"] == 0)).sum()
    ganancia = (TP * 273000) - (FP * 7000)
    return ganancia, TP, FP

# Inicializar estructuras para almacenar resultados
ganancias_semillas_original = {semilla: {corte: None for corte in cortes} for semilla in semillas}
ganancias_semillas_estacional = {semilla: {corte: None for corte in cortes} for semilla in semillas}

# Procesar cada corte para cada semilla
for semilla in semillas:
    print(f"Procesando semilla: {semilla}")
    for corte in cortes:
        print(f"Procesando corte: {corte}")
        file_original = os.path.join(input_dir, f"KA4210_{corte}_original_seed_{semilla}.csv")
        file_estacional = os.path.join(input_dir, f"KA4210_{corte}_estacional_seed_{semilla}.csv")

        if not os.path.exists(file_original) or not os.path.exists(file_estacional):
            print(f"Archivos no encontrados para corte {corte} y semilla {semilla}")
            continue

        predicciones_original = pd.read_csv(file_original)
        predicciones_estacional = pd.read_csv(file_estacional)

        pred_original = predicciones_original.merge(datos_reales, on="numero_de_cliente")
        pred_estacional = predicciones_estacional.merge(datos_reales, on="numero_de_cliente")

        ganancia_original, _, _ = calcular_ganancia_detallada(pred_original)
        ganancia_estacional, _, _ = calcular_ganancia_detallada(pred_estacional)

        ganancias_semillas_original[semilla][corte] = ganancia_original
        ganancias_semillas_estacional[semilla][corte] = ganancia_estacional

# Crear gráficos comparativos por semilla
for semilla in semillas:
    plt.figure(figsize=(10, 6))
    ganancias_original = [
        ganancias_semillas_original[semilla][corte] for corte in cortes if ganancias_semillas_original[semilla][corte] is not None
    ]
    ganancias_estacional = [
        ganancias_semillas_estacional[semilla][corte] for corte in cortes if ganancias_semillas_estacional[semilla][corte] is not None
    ]
    plt.plot(cortes[:len(ganancias_original)], ganancias_original, label=f"Original (Semilla {semilla})", marker="o", linestyle="--")
    plt.plot(cortes[:len(ganancias_estacional)], ganancias_estacional, label=f"Estacional (Semilla {semilla})", marker="o", linestyle="--")
    plt.xlabel("Corte")
    plt.ylabel("Ganancia")
    plt.title(f"Comparación de Ganancias por Corte - Semilla {semilla}")
    plt.legend()
    plt.grid(True)

    # Aplicar el formateador al eje y
    formatter = FuncFormatter(format_miles)
    plt.gca().yaxis.set_major_formatter(formatter)

    plt.tight_layout()
    plt.savefig(os.path.join(experimento_dir, f"ganancias_semilla_{semilla}.png"))
    plt.show()

# Calcular promedios por corte
promedios_original = {corte: sum(ganancias_semillas_original[semilla][corte] for semilla in semillas if ganancias_semillas_original[semilla][corte] is not None) / len([semilla for semilla in semillas if ganancias_semillas_original[semilla][corte] is not None]) for corte in cortes}
promedios_estacional = {corte: sum(ganancias_semillas_estacional[semilla][corte] for semilla in semillas if ganancias_semillas_estacional[semilla][corte] is not None) / len([semilla for semilla in semillas if ganancias_semillas_estacional[semilla][corte] is not None]) for corte in cortes}

# Gráfico promedio
plt.figure(figsize=(10, 6))
plt.plot(cortes, list(promedios_original.values()), label="Original (Promedio)", marker="o", linestyle="--", color="blue")
plt.plot(cortes, list(promedios_estacional.values()), label="Estacional (Promedio)", marker="o", linestyle="--", color="green")
plt.xlabel("Corte")
plt.ylabel("Ganancia Promedio")
plt.title("Comparación de Ganancias Promedio por Modelo y Corte")
plt.legend()
plt.grid(True)

# Aplicar el formateador al eje y
formatter = FuncFormatter(format_miles)
plt.gca().yaxis.set_major_formatter(formatter)

plt.tight_layout()
plt.savefig(os.path.join(experimento_dir, "ganancias_promedio.png"))
plt.show()
