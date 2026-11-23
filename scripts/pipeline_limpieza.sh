#!/usr/bin/env bash
# =============================================================================
# pipeline_limpieza.sh · HoopMind (dia 4, jueves 19/11)
#
#   raw (HDFS) -> ensuciar.py -> /nba/dirty -> Hive -> /nba/curated (Parquet)
#              -> comprobaciones + DESCRIBE FORMATTED + parquet-tools
#              -> verificar_limpieza.py (¿recupera la limpieza los originales?)
#
# Uso (desde la raiz del repo, con el Hadoop de hadoop/ levantado):
#   ./scripts/pipeline_limpieza.sh [fecha_raw] [semilla]
#   ./scripts/pipeline_limpieza.sh 2026-11-17 42
#
# Deja las evidencias en logs/ (cada paso con su .log).
# =============================================================================
set -euo pipefail
export MSYS_NO_PATHCONV=1          # Git Bash en Windows: no tocar las rutas /nba/...

FECHA=${1:-2026-11-17}
SEED=${2:-42}
DESDE=${DESDE:-0}                 # DESDE=7 salta los pasos 0-6 (ya hechos) y sigue en el 7
PY=${PY:-python3}                  # en Windows puede ser: PY=python ./scripts/...
TEMPORADA_MUESTRA=${TEMPORADA_MUESTRA:-2024}

RAW_HDFS="/nba/raw/fecha=$FECHA"
LOCAL_RAW="data/raw_$FECHA"
LOCAL_DIRTY="data/dirty"
EXPORT="data/curated_export"
HQL_DIR="/opt/hoopmind_hql"
BEELINE=(docker exec -i hive-server /opt/hive/bin/beeline -u jdbc:hive2://localhost:10000 -n root --silent=true)

paso() { echo; echo "=================== $* ==================="; }
hdfs() { docker exec namenode hdfs dfs "$@"; }
mkdir -p logs "$LOCAL_RAW" "$LOCAL_DIRTY" "$EXPORT" hive/generado

if [ "$DESDE" -lt 7 ]; then
paso "0. Contenedores"
for c in namenode datanode hive-metastore hive-server; do
  docker ps --format '{{.Names}}' | grep -qx "$c" || { echo "Falta el contenedor $c. Levanta el entorno: cd hadoop && docker compose up -d"; exit 1; }
done
hdfs -ls "$RAW_HDFS" | tee logs/00_raw_ls.log

paso "1. Copia local de los CSV originales (desde HDFS raw, no desde Kaggle)"
for f in Games.csv TeamStatistics.csv; do
  docker exec namenode bash -c "rm -f /tmp/$f && hdfs dfs -get $RAW_HDFS/$f /tmp/$f"
  docker cp "namenode:/tmp/$f" "$LOCAL_RAW/$f"
done
# De Extended solo hace falta la cabecera (no se ensucia)
docker exec namenode bash -c "hdfs dfs -cat $RAW_HDFS/TeamStatisticsExtended.csv 2>/dev/null | head -n 1" > "$LOCAL_RAW/TeamStatisticsExtended_cabecera.csv"
ls -l "$LOCAL_RAW"

paso "2. Ensuciado reproducible (semilla $SEED)"
"$PY" scripts/ensuciar.py --entrada "$LOCAL_RAW" --salida "$LOCAL_DIRTY" --seed "$SEED" | tee "logs/02_ensuciado_seed$SEED.log"

paso "3. Subida a HDFS: /nba/dirty y /nba/staging"
hdfs -mkdir -p /nba/dirty/games /nba/dirty/team_statistics /nba/dirty/log \
               /nba/staging/team_statistics_ext /nba/curated /tmp /user/hive/warehouse
docker exec namenode bash -c "rm -rf /tmp/dirty && mkdir -p /tmp/dirty"
for f in Games.csv TeamStatistics.csv ensuciado_log.csv ensuciado_resumen.json; do
  docker cp "$LOCAL_DIRTY/$f" "namenode:/tmp/dirty/$f"
done
hdfs -put -f /tmp/dirty/Games.csv /nba/dirty/games/
hdfs -put -f /tmp/dirty/TeamStatistics.csv /nba/dirty/team_statistics/
hdfs -put -f /tmp/dirty/ensuciado_log.csv /tmp/dirty/ensuciado_resumen.json /nba/dirty/log/
hdfs -cp -f "$RAW_HDFS/TeamStatisticsExtended.csv" /nba/staging/team_statistics_ext/
# curated se reescribe entera en cada ejecucion
docker exec namenode bash -c "hdfs dfs -rm -r -f -skipTrash '/nba/curated/*' >/dev/null 2>&1 || true"
hdfs -chmod -R 777 /nba/dirty /nba/staging /nba/curated /tmp /user/hive/warehouse
hdfs -du -h /nba/dirty /nba/staging | tee logs/03_hdfs_dirty_du.log

paso "4. DDL de entrada generado desde las cabeceras reales"
{
  echo "-- Generado por scripts/pipeline_limpieza.sh el $(date '+%F %T'). No editar a mano."
  echo "CREATE DATABASE IF NOT EXISTS nba_raw;"
  echo "CREATE DATABASE IF NOT EXISTS nba_dirty;"
  "$PY" scripts/ddl_desde_cabecera.py "$LOCAL_DIRTY/Games.csv" nba_dirty.games /nba/dirty/games
  "$PY" scripts/ddl_desde_cabecera.py "$LOCAL_DIRTY/TeamStatistics.csv" nba_dirty.team_statistics /nba/dirty/team_statistics
  "$PY" scripts/ddl_desde_cabecera.py "$LOCAL_RAW/TeamStatisticsExtended_cabecera.csv" nba_raw.team_statistics_ext /nba/staging/team_statistics_ext
} > hive/generado/01_tablas_entrada.hql
grep -c "STRING" hive/generado/01_tablas_entrada.hql | xargs echo "Columnas declaradas:"

paso "5. Hive: tablas de entrada, limpieza y escritura en Parquet"
docker exec hive-server bash -c "rm -rf $HQL_DIR && mkdir -p $HQL_DIR"
docker cp hive/. "hive-server:$HQL_DIR/"
"${BEELINE[@]}" -f "$HQL_DIR/generado/01_tablas_entrada.hql" 2>&1 | tee logs/05a_hive_tablas_entrada.log
"${BEELINE[@]}" -f "$HQL_DIR/02_limpieza_curated.hql"        2>&1 | tee logs/05b_hive_limpieza.log

paso "6. Comprobaciones y DESCRIBE FORMATTED"
"${BEELINE[@]}" --outputformat=table -f "$HQL_DIR/03_comprobaciones.hql" 2>&1 | tee logs/06_comprobaciones_hive.log
hdfs -du -h /nba/curated | tee logs/06_hdfs_curated_du.log

fi   # fin de los pasos 0-6

paso "7. Exportar curated a CSV (copiando los Parquet de HDFS, sin SELECT * por Hive)"
rm -rf data/curated_parquet && mkdir -p data/curated_parquet
for t in partidos estadisticas_equipo; do
  docker exec namenode bash -c "rm -rf /tmp/cur_$t && hdfs dfs -get /nba/curated/$t /tmp/cur_$t"
  docker cp "namenode:/tmp/cur_$t" "data/curated_parquet/$t"
done
"$PY" scripts/exportar_curated.py --origen data/curated_parquet --destino "$EXPORT" partidos estadisticas_equipo

paso "8. ¿Recupera la limpieza los valores originales?"
"$PY" scripts/verificar_limpieza.py --raw "$LOCAL_RAW" --log "$LOCAL_DIRTY/ensuciado_log.csv" \
      --curated "$EXPORT" --informe logs/08_verificacion_limpieza.csv | tee logs/08_verificacion_limpieza.log

paso "9. parquet-tools sobre un fichero real de /nba/curated"
for t in partidos estadisticas_equipo; do
  F=$(docker exec namenode bash -c "hdfs dfs -ls -C /nba/curated/$t/temporada=$TEMPORADA_MUESTRA | head -n 1")
  echo "Fichero: $F"
  docker exec namenode bash -c "rm -f /tmp/muestra_$t.parquet && hdfs dfs -get '$F' /tmp/muestra_$t.parquet"
  docker cp "namenode:/tmp/muestra_$t.parquet" "data/muestra_$t.parquet"
  if command -v parquet-tools >/dev/null 2>&1; then
    { parquet-tools inspect "data/muestra_$t.parquet"; parquet-tools show --head 5 "data/muestra_$t.parquet"; } \
      | tee "logs/09_parquet_tools_$t.log"
  else
    echo "parquet-tools no esta instalado. Instalalo con:  pip install parquet-tools"
    echo "y ejecuta:  parquet-tools inspect data/muestra_$t.parquet"
  fi
done

paso "10. Tabla de modelado compacta (04_modelado.hql) y comprobacion contra Python"
docker exec hive-server bash -c "mkdir -p $HQL_DIR" && docker cp hive/. "hive-server:$HQL_DIR/"
"${BEELINE[@]}" -f "$HQL_DIR/04_modelado.hql" 2>&1 | tee logs/10_hive_modelado.log
hdfs -du -h /nba/curated/modelado_partidos | tee logs/10_hdfs_modelado_du.log
docker exec namenode bash -c "rm -rf /tmp/cur_modelado_partidos && hdfs dfs -get /nba/curated/modelado_partidos /tmp/cur_modelado_partidos"
docker cp "namenode:/tmp/cur_modelado_partidos" "data/curated_parquet/modelado_partidos"
"$PY" scripts/exportar_curated.py --origen data/curated_parquet --destino "$EXPORT" modelado_partidos
"$PY" scripts/construir_modelado.py --curated "$EXPORT" --salida data/modelado_partidos.parquet | tee logs/10_modelado_python.log
"$PY" scripts/construir_modelado.py --curated "$EXPORT" --hive "$EXPORT/modelado_partidos.csv" | tee logs/10_comparacion_hive_python.log

paso "Hecho. Evidencias en logs/"
ls -1 logs/
