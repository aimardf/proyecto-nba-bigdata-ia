-- =============================================================================
-- 04_modelado.hql · HoopMind (Big Data + IA)
--
-- Construye la tabla de MODELADO compacta: nba_curated.modelado_partidos
-- (una fila por partido, 22 columnas en vez de las 55 de la primera version).
--
--   nba_curated.partidos + nba_curated.estadisticas_equipo
--        |  1. una fila por equipo y partido (dos por partido)      -> nba_work.largo
--        |  2. resumen de los 10 partidos ANTERIORES de cada equipo  -> nba_work.largo_hist
--        v  3. local - visitante, sumas y salidas                     -> nba_curated.modelado_partidos
--
-- Anti-fuga: las ventanas terminan en "1 PRECEDING" (el partido actual NO entra
-- en su propio resumen). Es el shift(1) + media de los ultimos 10 del notebook.
-- Equivale a scripts/construir_modelado.py; el pipeline compara los dos.
--
-- Columnas que NO pasan (y por que): ver docs/DICCIONARIO_CURATED.md, seccion
-- "modelado_partidos". Requiere haber ejecutado 02_limpieza_curated.hql.
-- =============================================================================

SET hive.exec.dynamic.partition = true;
SET hive.exec.dynamic.partition.mode = nonstrict;
SET hive.exec.max.dynamic.partitions = 1000;
SET hive.exec.max.dynamic.partitions.pernode = 1000;
SET parquet.compression = SNAPPY;

CREATE DATABASE IF NOT EXISTS nba_work;
CREATE DATABASE IF NOT EXISTS nba_curated;

-- ---------------------------------------------------------------------------
-- 1. Dos filas por partido (local y visitante). Solo temporada regular desde
--    2000-01 y partidos ya jugados. Las estadisticas de juego se pegan por
--    (game_id, team_id) desde estadisticas_equipo (ya limpias e imputadas).
-- ---------------------------------------------------------------------------
DROP TABLE IF EXISTS nba_work.largo;
CREATE TABLE nba_work.largo STORED AS PARQUET AS
SELECT
  b.game_id, b.game_date, b.temporada, b.equipo, b.es_local, b.pts_favor, b.pts_contra,
  CASE WHEN b.pts_favor > b.pts_contra THEN 1 ELSE 0 END AS gana,
  e.field_goals_percentage    AS tiro_campo_pct,
  e.three_pointers_percentage AS triples_pct,
  e.free_throws_percentage    AS libres_pct,
  e.rebounds_total            AS rebotes,
  e.assists                   AS asistencias,
  e.turnovers                 AS perdidas,
  e.steals                    AS robos,
  e.blocks                    AS tapones
FROM (
  SELECT game_id, game_date, temporada, home_team_id AS equipo, 1 AS es_local,
         home_score AS pts_favor, away_score AS pts_contra
    FROM nba_curated.partidos
   WHERE game_type = 'Regular Season' AND temporada >= 2000 AND jugado = 1
  UNION ALL
  SELECT game_id, game_date, temporada, away_team_id, 0,
         away_score, home_score
    FROM nba_curated.partidos
   WHERE game_type = 'Regular Season' AND temporada >= 2000 AND jugado = 1
) b
LEFT JOIN nba_curated.estadisticas_equipo e
  ON e.game_id = b.game_id AND e.team_id = b.equipo;

-- ---------------------------------------------------------------------------
-- 2. Resumen de los 10 partidos anteriores de cada equipo.
--    "10 PRECEDING AND 1 PRECEDING" = los 10 anteriores, sin el actual.
--    Solo vale si hay 10 datos (equivale a min_periods=10 de pandas).
-- ---------------------------------------------------------------------------
DROP TABLE IF EXISTS nba_work.largo_hist;
CREATE TABLE nba_work.largo_hist STORED AS PARQUET AS
SELECT
  game_id, game_date, temporada, equipo, es_local,
  CASE WHEN COUNT(pts_favor)      OVER w10 = 10 THEN AVG(pts_favor)      OVER w10 END AS pts_favor_ult10,
  CASE WHEN COUNT(pts_contra)     OVER w10 = 10 THEN AVG(pts_contra)     OVER w10 END AS pts_contra_ult10,
  CASE WHEN COUNT(gana)           OVER w10 = 10 THEN AVG(gana)           OVER w10 END AS win_pct_ult10,
  CASE WHEN COUNT(tiro_campo_pct) OVER w10 = 10 THEN AVG(tiro_campo_pct) OVER w10 END AS tiro_campo_pct_ult10,
  CASE WHEN COUNT(triples_pct)    OVER w10 = 10 THEN AVG(triples_pct)    OVER w10 END AS triples_pct_ult10,
  CASE WHEN COUNT(libres_pct)     OVER w10 = 10 THEN AVG(libres_pct)     OVER w10 END AS libres_pct_ult10,
  CASE WHEN COUNT(rebotes)        OVER w10 = 10 THEN AVG(rebotes)        OVER w10 END AS rebotes_ult10,
  CASE WHEN COUNT(asistencias)    OVER w10 = 10 THEN AVG(asistencias)    OVER w10 END AS asistencias_ult10,
  CASE WHEN COUNT(perdidas)       OVER w10 = 10 THEN AVG(perdidas)       OVER w10 END AS perdidas_ult10,
  CASE WHEN COUNT(robos)          OVER w10 = 10 THEN AVG(robos)          OVER w10 END AS robos_ult10,
  CASE WHEN COUNT(tapones)        OVER w10 = 10 THEN AVG(tapones)        OVER w10 END AS tapones_ult10,
  -- dias desde el partido anterior del equipo, con tope de 7
  LEAST(DATEDIFF(game_date, LAG(game_date) OVER (PARTITION BY equipo ORDER BY game_date, game_id)), 7) AS dias_descanso,
  -- % de victorias de la temporada ANTES de este partido
  CASE WHEN COUNT(*) OVER wtemp > 0
       THEN COALESCE(SUM(gana) OVER wtemp, 0) / COUNT(*) OVER wtemp END AS win_pct_temporada
FROM nba_work.largo
WINDOW
  w10   AS (PARTITION BY equipo ORDER BY game_date, game_id ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING),
  wtemp AS (PARTITION BY equipo, temporada ORDER BY game_date, game_id ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING);

-- ---------------------------------------------------------------------------
-- 3. Una fila por partido: diferencia local - visitante y salidas.
--    Solo se quedan los partidos donde los dos equipos ya tienen 10 previos.
-- ---------------------------------------------------------------------------
DROP TABLE IF EXISTS nba_curated.modelado_partidos;
CREATE EXTERNAL TABLE nba_curated.modelado_partidos (
  game_id BIGINT, game_date DATE, home_team_id BIGINT, away_team_id BIGINT,
  dif_pts_favor_ult10 DOUBLE, dif_pts_contra_ult10 DOUBLE, dif_win_pct_ult10 DOUBLE,
  dif_tiro_campo_pct_ult10 DOUBLE, dif_triples_pct_ult10 DOUBLE, dif_libres_pct_ult10 DOUBLE,
  dif_rebotes_ult10 DOUBLE, dif_asistencias_ult10 DOUBLE, dif_perdidas_ult10 DOUBLE,
  dif_robos_ult10 DOUBLE, dif_tapones_ult10 DOUBLE,
  dif_dias_descanso DOUBLE, dif_win_pct_temporada DOUBLE,
  suma_pts_favor_ult10 DOUBLE, suma_pts_contra_ult10 DOUBLE,
  home_win INT, puntos_totales INT
)
PARTITIONED BY (temporada INT)
STORED AS PARQUET
LOCATION '/nba/curated/modelado_partidos'
TBLPROPERTIES ("parquet.compression" = "SNAPPY");

INSERT OVERWRITE TABLE nba_curated.modelado_partidos PARTITION (temporada)
SELECT
  p.game_id, p.game_date, p.home_team_id, p.away_team_id,
  l.pts_favor_ult10      - v.pts_favor_ult10,
  l.pts_contra_ult10     - v.pts_contra_ult10,
  l.win_pct_ult10        - v.win_pct_ult10,
  l.tiro_campo_pct_ult10 - v.tiro_campo_pct_ult10,
  l.triples_pct_ult10    - v.triples_pct_ult10,
  l.libres_pct_ult10     - v.libres_pct_ult10,
  l.rebotes_ult10        - v.rebotes_ult10,
  l.asistencias_ult10    - v.asistencias_ult10,
  l.perdidas_ult10       - v.perdidas_ult10,
  l.robos_ult10          - v.robos_ult10,
  l.tapones_ult10        - v.tapones_ult10,
  l.dias_descanso        - v.dias_descanso,
  l.win_pct_temporada    - v.win_pct_temporada,
  l.pts_favor_ult10  + v.pts_favor_ult10,
  l.pts_contra_ult10 + v.pts_contra_ult10,
  CASE WHEN p.home_score > p.away_score THEN 1 ELSE 0 END,   -- objetivo 1
  CAST(p.home_score + p.away_score AS INT),                  -- objetivo 2
  p.temporada
FROM nba_curated.partidos p
JOIN nba_work.largo_hist l ON l.game_id = p.game_id AND l.es_local = 1
JOIN nba_work.largo_hist v ON v.game_id = p.game_id AND v.es_local = 0
WHERE p.game_type = 'Regular Season' AND p.temporada >= 2000 AND p.jugado = 1
  AND l.pts_favor_ult10 IS NOT NULL AND v.pts_favor_ult10 IS NOT NULL;

-- Comprobaciones rapidas
SELECT COUNT(*) AS partidos, COUNT(DISTINCT game_id) AS ids_distintos FROM nba_curated.modelado_partidos;
SELECT temporada, COUNT(*) AS partidos, ROUND(AVG(home_win), 3) AS pct_gana_local,
       ROUND(AVG(puntos_totales), 1) AS media_puntos
FROM nba_curated.modelado_partidos GROUP BY temporada ORDER BY temporada;
DESCRIBE FORMATTED nba_curated.modelado_partidos;
