# Validación cruzada temporal (TimeSeriesSplit)

**Qué es.** Un examen repetido 5 veces, siempre entrenando con el pasado y examinando con el bloque siguiente.
Mezclar filas al azar (KFold normal) sería hacer trampa: el modelo vería partidos de 2020 al entrenar y le pediríamos predecir 2015.

**Qué datos usa.** Solo 2000-01 a 2022-23. La temporada 2023-24 sigue siendo el test final y 2024-25 en adelante el lote 2 (drift). No se tocan.

**Qué mide.** Objetivo 1: accuracy y ROC-AUC. Objetivo 2: MAE, RMSE y sesgo (predicho − real), con `*_relativo` frente a `*_absoluto`.
La media y la desviación de los 5 cortes dicen cuánto se puede fiar uno de la nota: desviación baja = resultado estable.

**Por qué el absoluto debe salir peor en cortes recientes.** Los puntos por partido suben con los años; un modelo absoluto entrenado con el pasado se queda corto (sesgo negativo cada vez mayor). El relativo resta la media de liga reciente, así que su sesgo se mantiene cerca de 0.

**Ejecutar** (en el PC de Aimar, desde la raíz del repo):

    python3 scripts/validacion_temporal.py --tabla data/modelado_partidos.parquet
    mlflow ui --backend-store-uri sqlite:///mlflow.db     # experimentos hoopmind_obj1_home_win y hoopmind_obj2_puntos_totales, runs cv_*

Salidas en `docs/validacion_temporal/`: `cortes.csv`, `metricas_por_corte.csv`, `resumen_cv.csv`, `relativo_vs_absoluto_por_corte.png`.
Probar sin datos reales: añadir `--sintetico`.
