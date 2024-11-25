Para correr el experimento:

1) Ejecutar j024_tendencia.py para ver la tendencia a lo largo del tiempo de cada variable.
2) Ejecutar j024_variablesestacionales.py para agregar variables de estacionalidad al dataset.
3) Ejecutar j024_comprobarestacionalidad.py para dejar el archivo con estacionalidad comparable en foto_mes al original.
4) Ejecutar 422 y 421 en R, que calculan un lightgbm en cada dataset
5) Ejecutar j024_comparacion.py que comparará la ganancia en cada punto de corte, en cada dataset.