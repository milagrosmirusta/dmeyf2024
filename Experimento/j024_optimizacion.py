import os
import pandas as pd
import lightgbm as lgb
import numpy as np
import optuna
from sklearn.metrics import log_loss

# Definir los parámetros
PARAM = {
    "experimento": "LGBMEst",
    "semilla_primigenia": 173717,
    "num_seeds": 5,
    "input": {
        "dataset_original": "~/buckets/b1/datasets/competencia_02.csv.gz",
        "dataset_estacional": "~/buckets/b1/datasets/dmeyf2024/Experimento/dataset_con_variables_estacionales.csv.gz",
        "training_months": [202012, 202011, 202010, 202009, 202008],
        "validation_month": 202006,
        "future_month": 202106
    },
}

# Crear el directorio de salida
output_dir = f"/home/mili_irusta/buckets/b1/datasets/dmeyf2024/Experimento/exp/{PARAM['experimento']}"
os.makedirs(output_dir, exist_ok=True)

# Función para preparar el dataset
def preparar_dataset(filepath, convert_foto_mes=False):
    dataset = pd.read_csv(filepath, compression='gzip')
    if convert_foto_mes:
        dataset['foto_mes'] = pd.to_datetime(dataset['foto_mes'], errors='coerce').dt.strftime('%Y%m').astype(int)
    else:
        dataset['foto_mes'] = dataset['foto_mes'].astype(int)
    dataset['clase01'] = np.where(dataset['clase_ternaria'].isin(["BAJA+1", "BAJA+2"]), 1, 0)
    return dataset

# Cargar los datasets
dataset_original = preparar_dataset(PARAM["input"]["dataset_original"])
dataset_estacional = preparar_dataset(PARAM["input"]["dataset_estacional"], convert_foto_mes=True)

# Campos para entrenamiento
campos_buenos = [col for col in dataset_original.columns if col not in ["clase_ternaria", "clase01", "foto_mes"]]

# Dividir el dataset original en entrenamiento y validación
train_data = dataset_original[dataset_original['foto_mes'].isin(PARAM["input"]["training_months"])]
val_data = dataset_original[dataset_original['foto_mes'] == PARAM["input"]["validation_month"]]

X_train, y_train = train_data[campos_buenos], train_data['clase01']
X_val, y_val = val_data[campos_buenos], val_data['clase01']

# Lista para almacenar los resultados de cada prueba
resultados_optimizacion = []

# Función objetivo para optimización con Optuna
def objective(trial):
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 300),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 1000),
         "num_iterations": trial.suggest_int("num_iterations", 100, 1000),  # Agregado
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 5.0),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "max_bin": trial.suggest_int("max_bin", 15, 255),
        "seed": PARAM["semilla_primigenia"]
    }
    
    # Crear dataset para LightGBM
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    # Entrenar el modelo con early stopping y sin mensajes de evaluación
    model = lgb.train(
        params,
        dtrain,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=0)]
    )
    
    # Calcular log_loss en el conjunto de validación
    y_pred = model.predict(X_val, num_iteration=model.best_iteration)
    loss = log_loss(y_val, y_pred)
    
    # Guardar el resultado de este trial en la lista
    resultados_optimizacion.append({"trial": trial.number, "params": params, "log_loss": loss})
    
    return loss

# Ejecutar optimización con Optuna
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

# Guardar los resultados detallados de cada prueba en un archivo CSV
resultados_df = pd.DataFrame(resultados_optimizacion)
resultados_df.to_csv(f"{output_dir}/resultados_optimizacion.csv", index=False)
print("Resultados de optimización guardados en resultados_optimizacion.csv")

# Obtener los mejores parámetros y guardarlos
best_params = study.best_params
print(f"Mejores hiperparámetros: {best_params}")
with open(f"{output_dir}/mejores_parametros.txt", "w") as f:
    f.write(str(best_params))

