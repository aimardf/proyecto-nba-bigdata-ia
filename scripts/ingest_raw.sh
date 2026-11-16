#!/bin/bash
# Uso: ./ingest_raw.sh <dataset_kaggle> [fecha]
set -e
DS=$1; FECHA=${2:-$(date +%F)}
mkdir -p data/raw logs
kaggle datasets download -d "$DS" -p data/raw --unzip
docker exec namenode hdfs dfs -mkdir -p /nba/raw/fecha=$FECHA
docker cp data/raw/. namenode:/tmp/raw_$FECHA
docker exec namenode bash -c "hdfs dfs -put -f /tmp/raw_$FECHA/*.csv /nba/raw/fecha=$FECHA/"
docker exec namenode hdfs dfs -du -h /nba/raw/fecha=$FECHA | tee logs/ingesta_$FECHA.log
