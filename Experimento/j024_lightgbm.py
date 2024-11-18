import os
import pandas as pd
import lightgbm as lgb
import numpy as np
import ast

# Definir los parámetros base
PARAM = {
    "experimento": "LGBMEst",
    "semilla_primigenia": 173717,
    "num_seeds": 5,
    "input": {
        "dataset_original": "~/buckets/b1/datasets/competencia_02.csv.gz",
        "dataset_estacional": "~/buckets/b1/datasets/dmeyf2024/Experimento/dataset_con_variables_estacionales_limpio.csv.gz",
        "training_months": [202012, 202011, 202010, 202009, 202008],
        "validation_month": 202006,
        "future_month": 202106
    },
    "finalmodel": {}  # Inicializar vacío para luego completarlo
}

# Ruta al directorio de salida y archivo de hiperparámetros
output_dir = f"/home/mili_irusta/buckets/b1/datasets/dmeyf2024/Experimento/exp/{PARAM['experimento']}"
os.makedirs(output_dir, exist_ok=True)
hyperparams_file = f"{output_dir}/mejores_parametros.txt"

# Función para leer hiperparámetros desde el archivo txt y actualizar PARAM["finalmodel"]
def cargar_hiperparametros(filepath):
    with open(filepath, "r") as f:
        hyperparams_str = f.read()
    # Convertir el string del archivo en un diccionario
    hyperparams = ast.literal_eval(hyperparams_str)
    PARAM["finalmodel"].update(hyperparams)
    print("Hiperparámetros cargados:", PARAM["finalmodel"])  # Validación

# Cargar los hiperparámetros optimizados
cargar_hiperparametros(hyperparams_file)

# Función para preparar el dataset y unificar el formato de 'foto_mes'
def preparar_dataset(filepath):
    dataset = pd.read_csv(filepath, compression='gzip')
    print(f"Dataset cargado desde {filepath}. Dimensiones: {dataset.shape}")

    # Comprobar si 'foto_mes' existe en las columnas
    if 'foto_mes' not in dataset.columns:
        raise ValueError(f"El archivo {filepath} no contiene la columna 'foto_mes'.")
    
    # Manejar diferentes formatos de 'foto_mes'
    if pd.api.types.is_datetime64_any_dtype(dataset['foto_mes']):
        dataset['foto_mes'] = dataset['foto_mes'].dt.strftime('%Y%m').astype(int)
    elif dataset['foto_mes'].dtype == object:
        # Intentar convertir a datetime y luego a formato numérico
        dataset['foto_mes'] = pd.to_datetime(dataset['foto_mes'], format='%Y%m', errors='coerce')
        # Contar valores no convertidos
        no_convertidos = dataset['foto_mes'].isna().sum()
        if no_convertidos > 0:
            print(f"Advertencia: {no_convertidos} valores en 'foto_mes' no se pudieron convertir y serán eliminados.")
        dataset = dataset.dropna(subset=['foto_mes'])
        dataset['foto_mes'] = dataset['foto_mes'].dt.strftime('%Y%m').astype(int)
    elif dataset['foto_mes'].dtype in [int, float]:
        # Asumir que ya está en formato adecuado, pero eliminar NaNs
        dataset = dataset.dropna(subset=['foto_mes'])
        dataset['foto_mes'] = dataset['foto_mes'].astype(int)
    else:
        raise ValueError("El formato de la columna 'foto_mes' no es compatible.")
    
    print("Valores únicos en foto_mes después de conversión:", dataset['foto_mes'].unique())

    # Crear la columna 'clase01'
    dataset['clase01'] = np.where(dataset['clase_ternaria'].isin(["BAJA+1", "BAJA+2"]), 1, 0)
    return dataset



# Cargar los datasets
dataset_original = preparar_dataset(PARAM["input"]["dataset_original"])
dataset_estacional = preparar_dataset(PARAM["input"]["dataset_estacional"])

# Definir campos para entrenamiento (excluyendo la clase y 'foto_mes')
campos_buenos = [col for col in dataset_original.columns if col not in ["clase_ternaria", "clase01", "foto_mes"]]
print("Campos buenos definidos:", campos_buenos)  # Validación

# Función para entrenar y evaluar el modelo
def entrenar_y_evaluar(dataset, seed):
    # Filtrar meses de entrenamiento y futuro mes de predicción
    train_data = dataset[dataset['foto_mes'].isin(PARAM["input"]["training_months"])]
    future_data = dataset[dataset['foto_mes'] == PARAM["input"]["future_month"]]
    
    print(f"Tamaño de train_data con seed {seed}: {train_data.shape}")  # Validación
    print(train_data.head())  # Validación
    print(f"Tamaño de future_data con seed {seed}: {future_data.shape}")  # Validación
    print(future_data.head())  # Validación

    # Validaciones explícitas para conjuntos vacíos
    if train_data.empty:
        raise ValueError(f"train_data está vacío para seed {seed}. Verifica los filtros de training_months.")
    if future_data.empty:
        raise ValueError(f"future_data está vacío para el mes futuro {PARAM['input']['future_month']}.")

    # Preparar los datos para LightGBM
    dtrain = lgb.Dataset(data=train_data[campos_buenos], label=train_data['clase01'])

    # Configurar parámetros del modelo con los hiperparámetros optimizados
    lgb_params = {
        "objective": "binary",
        "learning_rate": PARAM["finalmodel"]["learning_rate"],
        "num_leaves": PARAM["finalmodel"]["num_leaves"],
        "min_data_in_leaf": PARAM["finalmodel"]["min_data_in_leaf"],
        "feature_fraction": PARAM["finalmodel"]["feature_fraction"],
        "max_bin": PARAM["finalmodel"]["max_bin"],
        "num_iterations": PARAM["finalmodel"]["num_iterations"],
        "seed": seed
    }

    print("Parámetros para LightGBM:", lgb_params)  # Validación

    # Entrenar el modelo
    modelo = lgb.train(lgb_params, dtrain)

    # Aplicar el modelo en los datos futuros (junio 2021)
    prediccion = modelo.predict(future_data[campos_buenos])

    # Crear tabla de resultados
    tb_entrega = future_data[['numero_de_cliente', 'foto_mes', 'clase01']].copy()
    tb_entrega['prob'] = prediccion

    # Evaluación de la ganancia en diferentes cortes
    resultados = []
    cortes = np.arange(9000, 13001, 500)
    for corte in cortes:
        tb_entrega['Predicted'] = (tb_entrega['prob'].rank(method="first", ascending=False) <= corte).astype(int)
        ganancia = calcular_ganancia(tb_entrega)
        resultados.append({"seed": seed, "corte": corte, "ganancia": ganancia})
    
    return resultados

# Función para calcular la ganancia
def calcular_ganancia(tb_entrega):
    tp = tb_entrega[(tb_entrega['clase01'] == 1) & (tb_entrega['Predicted'] == 1)].shape[0]
    fp = tb_entrega[(tb_entrega['clase01'] == 0) & (tb_entrega['Predicted'] == 1)].shape[0]
    return tp * 7800 - fp * 2000

# Correr el experimento para ambas versiones del dataset y múltiples semillas
resultados_totales = []
for seed in range(1, PARAM["num_seeds"] + 1):
    print(f"Ejecutando con semilla {seed} para el dataset original...")
    resultados_original = entrenar_y_evaluar(dataset_original, seed)
    resultados_totales.extend([{"dataset": "original", **res} for res in resultados_original])
    
    print(f"Ejecutando con semilla {seed} para el dataset con variables estacionales...")
    resultados_estacional = entrenar_y_evaluar(dataset_estacional, seed)
    resultados_totales.extend([{"dataset": "estacional", **res} for res in resultados_estacional])

# Guardar los resultados en un archivo CSV
resultados_df = pd.DataFrame(resultados_totales)
resultados_df.to_csv(f"{output_dir}/resultados_ganancia.csv", index=False)
print("Resultados guardados en resultados_ganancia.csv")
