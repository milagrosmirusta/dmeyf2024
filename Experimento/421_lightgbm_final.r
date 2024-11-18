# para correr el Google Cloud

# limpio la memoria
rm(list = ls()) # remove all objects
gc() # garbage collection

require("data.table")
require("lightgbm")

# defino los parametros de la corrida, en una lista, la variable global  PARAM
PARAM <- list()
PARAM$experimento <- "KA4210"
PARAM$semillas <- c(173717, 481129, 839203, 761081, 594017) # Lista de semillas
PARAM$dataset_type <- "estacional"  # Puede ser "original" o "estacional"

PARAM$input$dataset <- "~/buckets/b1/datasets/dmeyf2024/Experimento/dataset_con_variables_estacionales_limpio.csv.gz"
PARAM$input$training <- c(202012, 202011, 202010, 202009, 202008) # meses donde se entrena el modelo
PARAM$input$future <- c(202106) # meses donde se aplica el modelo

PARAM$finalmodel$num_iterations <- 3145
PARAM$finalmodel$learning_rate <- 0.0347900153108068
PARAM$finalmodel$feature_fraction <- 0.551934348639222
PARAM$finalmodel$min_data_in_leaf <- 987
PARAM$finalmodel$num_leaves <- 732
PARAM$finalmodel$max_bin <- 31

#------------------------------------------------------------------------------
# Aqui empieza el programa
setwd("~/buckets/b1")

# cargo el dataset donde voy a entrenar
dataset <- fread(PARAM$input$dataset, stringsAsFactors = TRUE)

#--------------------------------------

# paso la clase a binaria que tome valores {0,1}  enteros
dataset[, clase01 := ifelse(clase_ternaria %in% c("BAJA+2", "BAJA+1"), 1L, 0L)]

#--------------------------------------

# los campos que se van a utilizar
campos_buenos <- setdiff(colnames(dataset), c("clase_ternaria", "clase01"))

#--------------------------------------

# establezco donde entreno
dataset[, train := 0L]
dataset[foto_mes %in% PARAM$input$training, train := 1L]

#--------------------------------------
# calcular scale_pos_weight para desbalance
num_negativos <- nrow(dataset[train == 1L & clase01 == 0L])
num_positivos <- nrow(dataset[train == 1L & clase01 == 1L])
scale_pos_weight <- num_negativos / num_positivos

#--------------------------------------
# creo las carpetas donde van los resultados
dir.create("./exp/", showWarnings = FALSE)
dir.create(paste0("./exp/", PARAM$experimento, "/"), showWarnings = FALSE)

# Establezco el Working Directory DEL EXPERIMENTO
setwd(paste0("./exp/", PARAM$experimento, "/"))

#--------------------------------------
# Entrenamiento y generación de predicciones para cada semilla
for (semilla in PARAM$semillas) {
  cat("\nEntrenando con semilla:", semilla, "\n")
  
  # Dejo los datos en el formato que necesita LightGBM
  dtrain <- lgb.Dataset(
    data = data.matrix(dataset[train == 1L, campos_buenos, with = FALSE]),
    label = dataset[train == 1L, clase01]
  )
  
  # genero el modelo
  modelo <- lgb.train(
    data = dtrain,
    param = list(
      objective = "binary",
      max_bin = PARAM$finalmodel$max_bin,
      learning_rate = PARAM$finalmodel$learning_rate,
      num_iterations = PARAM$finalmodel$num_iterations,
      num_leaves = PARAM$finalmodel$num_leaves,
      min_data_in_leaf = PARAM$finalmodel$min_data_in_leaf,
      feature_fraction = PARAM$finalmodel$feature_fraction,
      seed = semilla,  # Usar la semilla actual
      scale_pos_weight = scale_pos_weight  # Manejo de desbalance
    )
  )
  
  #--------------------------------------
  # ahora imprimo la importancia de variables
  tb_importancia <- as.data.table(lgb.importance(modelo))
  archivo_importancia <- paste0("impo_", PARAM$dataset_type, "_seed_", semilla, ".txt")
  fwrite(tb_importancia, file = archivo_importancia, sep = "\t")
  
  #--------------------------------------
  # grabo a disco el modelo en un formato para seres humanos
  lgb.save(modelo, paste0("modelo_", PARAM$dataset_type, "_seed_", semilla, ".txt"))
  
  #--------------------------------------
  # aplico el modelo a los datos sin clase
  dapply <- dataset[foto_mes == PARAM$input$future]
  
  # aplico el modelo a los datos nuevos
  prediccion <- predict(
    modelo,
    data.matrix(dapply[, campos_buenos, with = FALSE])
  )
  
  # genero la tabla de entrega
  tb_entrega <- dapply[, list(numero_de_cliente, foto_mes)]
  tb_entrega[, prob := prediccion]
  
  # grabo las probabilidades del modelo
  fwrite(tb_entrega, file = paste0("prediccion_", PARAM$dataset_type, "_seed_", semilla, ".txt"), sep = "\t")
  
  # ordeno por probabilidad descendente
  setorder(tb_entrega, -prob)
  
  # genero archivos con los "envios" mejores
  cortes <- seq(9000, 13000, by = 500)
  for (envios in cortes) {
    tb_entrega[, Predicted := 0L]
    tb_entrega[1:envios, Predicted := 1L]
    
    # Incluir el tipo de dataset y la semilla en el nombre del archivo
    fwrite(tb_entrega[, list(numero_de_cliente, Predicted)],
           file = paste0(PARAM$experimento, "_", envios, "_", PARAM$dataset_type, "_seed_", semilla, ".csv"),
           sep = ","
    )
  }
}

cat("\n\nLa generación de los archivos para Kaggle ha terminado\n")