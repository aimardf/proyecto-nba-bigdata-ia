# Proyecto NBA · Big Data + IA

Predicción del ganador de un partido (clasificación) y de los puntos totales del partido (regresión) con datos de la NBA a nivel de equipo y partido (sin datos de jugadores), usando Hadoop (HDFS + Hive, sin PySpark), MLflow y FastAPI.

## Estructura
- `hadoop/`: Docker Compose con HDFS, YARN y Hive.
- `scripts/`: ingesta a HDFS (`ingest_raw.sh`) y ensuciado reproducible (`ensuciar.py`).
- `notebooks/`: EDA y modelos.
- `api/`: API FastAPI.
- `docs/`: informe, diagramas, capturas de evidencias.

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

## Estado
Entrega 1 en curso. Ver `docs/`.
