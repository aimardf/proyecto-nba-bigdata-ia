# Proyecto NBA · Big Data + IA (HoopMind)

Predicción del ganador de un partido (clasificación) y de los puntos totales del partido (regresión) con datos de la NBA a nivel de equipo y partido (sin datos de jugadores), usando Hadoop (HDFS + Hive, sin PySpark), MLflow y FastAPI.

## Estructura
- `hadoop/`: Docker Compose con HDFS y Hive (sin YARN: Hive ejecuta sus trabajos en modo local dentro de `hive-server`).
- `scripts/`: ingesta (`ingest_raw.sh`), ensuciado (`ensuciar.py`), pipeline de limpieza (`pipeline_limpieza.sh`) y verificación (`verificar_limpieza.py`).
- `hive/`: limpieza con HiveQL (`02_limpieza_curated.hql`) y comprobaciones (`03_comprobaciones.hql`). `hive/generado/` lo crea el pipeline.
- `notebooks/`: EDA y modelos.
- `api/`: API FastAPI.
- `docs/`: informe, diagramas, diccionario de datos (`DICCIONARIO_CURATED.md`) y capturas.

## Puesta en marcha (Hadoop)
```bash
cd hadoop && docker compose up -d
docker exec namenode hdfs dfs -ls /      # evidencia: HDFS operativo
```

## Ingesta en bruto
Requiere `~/.kaggle/kaggle.json`. Importante: no se suben a HDFS los ficheros de jugadores ni de árbitros (minimización de datos).
```bash
./scripts/ingest_raw.sh eoinamoore/historical-nba-data-and-player-box-scores 2026-11-17
```

## Ensuciado, limpieza con Hive y Parquet
Un solo comando desde la raíz del repo (Python 3 con pandas en el ordenador):
```bash
./scripts/pipeline_limpieza.sh 2026-11-17 42          # fecha de raw y semilla
# En Windows con Git Bash, si "python3" no existe:  PY=python ./scripts/pipeline_limpieza.sh 2026-11-17 42
```
Qué hace:
1. Baja de `/nba/raw/fecha=2026-11-17` los CSV de Games y TeamStatistics.
2. `ensuciar.py --seed 42`: 8 problemas reproducibles + quita `officials` y `coachId`. Deja `ensuciado_log.csv` con cada celda cambiada.
3. Sube lo ensuciado a `/nba/dirty` y TeamStatisticsExtended a `/nba/staging`.
4. Crea las tablas de Hive leyendo las cabeceras reales (todo STRING).
5. Limpia con Hive y escribe `/nba/curated/{partidos, estadisticas_equipo, estadisticas_avanzadas}` en Parquet + Snappy, particionado por temporada.
6. Comprobaciones, `DESCRIBE FORMATTED` y `SHOW PARTITIONS`.
7. Exporta curated a CSV y `verificar_limpieza.py` mide cuánto de lo ensuciado se recupera y si se ha estropeado algo que no estaba sucio.
8. `parquet-tools inspect` sobre un fichero real (`pip install parquet-tools`).

Todas las evidencias quedan en `logs/` (un `.log` por paso).

## Tabla de modelado compacta y modelos base (23/11)
```bash
python3 scripts/construir_modelado.py --curated data/curated_export     # tabla compacta (22 columnas)
python3 scripts/comparar_columnas.py --datos ~/Desktop/nba-datos        # por qué 15 entradas y no 48
python3 scripts/modelos_base.py --tabla data/modelado_partidos.parquet  # EDA + modelos base con MLflow
mlflow ui --backend-store-uri sqlite:///mlflow.db                       # http://127.0.0.1:5000
```
En Hive, `hive/04_modelado.hql` crea `nba_curated.modelado_partidos` (el pipeline lo ejecuta en el paso 10 y lo compara con Python).

## Estado
Entrega 1 hecha (18/11). Día 4: ensuciado, limpieza con Hive, Parquet y diccionario.
