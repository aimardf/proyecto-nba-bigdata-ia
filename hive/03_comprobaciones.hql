-- =============================================================================
-- 03_comprobaciones.hql · HoopMind
-- Evidencias de la limpieza para el informe (Entrega 2). Lo ejecuta el pipeline y
-- guarda la salida en logs/comprobaciones_hive.log.
-- =============================================================================

-- 1. Filas: dirty -> work (sin duplicados) -> curated
SELECT * FROM (
  SELECT '1 games / dirty' AS paso, COUNT(*) AS filas FROM nba_dirty.games
  UNION ALL SELECT '2 games / sin duplicados', COUNT(*) FROM nba_work.games_tipado
  UNION ALL SELECT '3 games / curated.partidos', COUNT(*) FROM nba_curated.partidos
  UNION ALL SELECT '4 team_statistics / dirty', COUNT(*) FROM nba_dirty.team_statistics
  UNION ALL SELECT '5 team_statistics / sin duplicados', COUNT(*) FROM nba_work.ts_tipado
  UNION ALL SELECT '6 team_statistics / curated.estadisticas_equipo', COUNT(*) FROM nba_curated.estadisticas_equipo
  UNION ALL SELECT '7 team_statistics_ext / raw', COUNT(*) FROM nba_raw.team_statistics_ext
  UNION ALL SELECT '8 team_statistics_ext / curated.estadisticas_avanzadas', COUNT(*) FROM nba_curated.estadisticas_avanzadas
) f ORDER BY paso;

-- 2. Duplicados que quedan en curated (tiene que salir 0)
SELECT 'partidos' AS tabla, COUNT(*) AS claves_repetidas
FROM (SELECT game_id FROM nba_curated.partidos GROUP BY game_id HAVING COUNT(*) > 1) d
UNION ALL
SELECT 'estadisticas_equipo', COUNT(*)
FROM (SELECT game_id, team_id FROM nba_curated.estadisticas_equipo GROUP BY game_id, team_id HAVING COUNT(*) > 1) d;

-- 3. Fechas que no se han podido leer (se quedan fuera de curated)
SELECT 'games' AS tabla, SUM(sin_fecha) AS sin_fecha FROM nba_work.games_tipado
UNION ALL
SELECT 'team_statistics', SUM(sin_fecha) FROM nba_work.ts_tipado;

-- 4. La temporada sacada de la fecha coincide con la del gameId (digitos 2-3)
SELECT COUNT(*) AS partidos, SUM(CASE WHEN t_id <> temporada THEN 1 ELSE 0 END) AS no_coinciden
FROM (
  SELECT temporada,
         CASE WHEN pmod(CAST(floor(game_id / 100000) AS INT), 100) < 46
              THEN 2000 + pmod(CAST(floor(game_id / 100000) AS INT), 100)
              ELSE 1900 + pmod(CAST(floor(game_id / 100000) AS INT), 100) END AS t_id
  FROM nba_curated.partidos
) x;

-- 5. Que ha corregido la limpieza (marcas)
SELECT SUM(marca_intercambio) AS local_visitante_corregidos,
       SUM(marca_marcador)    AS marcadores_corregidos,
       SUM(marca_ganador_incoherente) AS ganador_incoherente,
       SUM(jugado) AS jugados, SUM(1 - jugado) AS sin_jugar
FROM nba_curated.partidos;

SELECT SUM(CASE WHEN marca_fuera_rango > 0 THEN 1 ELSE 0 END) AS filas_con_valores_fuera_de_rango,
       SUM(marca_score_recalculado)  AS puntos_recalculados,
       SUM(marca_score_incoherente)  AS puntos_distintos_de_suma_de_cuartos,
       SUM(n_recuperados_regla)      AS datos_recuperados_con_reglas,
       SUM(n_huecos_antes_de_imputar) AS huecos_antes_de_imputar
FROM nba_curated.estadisticas_equipo;

-- 6. Partidos y filas de equipo: cada partido jugado deberia tener 2 filas
SELECT n_filas_equipo, COUNT(*) AS partidos
FROM (
  SELECT p.game_id, COUNT(e.team_id) AS n_filas_equipo
  FROM nba_curated.partidos p
  LEFT JOIN nba_curated.estadisticas_equipo e ON e.game_id = p.game_id
  WHERE p.jugado = 1
  GROUP BY p.game_id
) x
GROUP BY n_filas_equipo;

-- Filas de equipo cuyo partido no esta en Games (la diferencia de wc -l del 17/11)
SELECT COUNT(DISTINCT e.game_id) AS partidos_solo_en_team_statistics
FROM nba_curated.estadisticas_equipo e
LEFT JOIN nba_curated.partidos p ON p.game_id = e.game_id
WHERE p.game_id IS NULL;

-- 7. Tipos de partido (para decidir que se excluye: pretemporada, All-Star...)
SELECT game_type, COUNT(*) AS partidos, MIN(temporada) AS desde, MAX(temporada) AS hasta
FROM nba_curated.partidos GROUP BY game_type ORDER BY partidos DESC;

-- 8. Nulos que quedan en temporada regular desde 2000-01 (lo que usara el modelo)
SELECT COUNT(*) AS filas,
       SUM(CASE WHEN team_score IS NULL THEN 1 ELSE 0 END)             AS team_score_nulo,
       SUM(CASE WHEN field_goals_percentage IS NULL THEN 1 ELSE 0 END) AS fg_pct_nulo,
       SUM(CASE WHEN rebounds_total IS NULL THEN 1 ELSE 0 END)         AS rebotes_nulo,
       SUM(CASE WHEN assists IS NULL THEN 1 ELSE 0 END)                AS asistencias_nulo,
       SUM(CASE WHEN turnovers IS NULL THEN 1 ELSE 0 END)              AS perdidas_nulo
FROM nba_curated.estadisticas_equipo
WHERE temporada >= 2000 AND game_type = 'Regular Season';

-- 9. Detector de anomalias (idea A1): puntos totales fuera de Q1 - 1,5 IQR y Q3 + 1,5 IQR
--    de su temporada. No se borran: se cuentan para el EDA.
SELECT p.temporada, COUNT(*) AS partidos,
       SUM(CASE WHEN p.puntos_totales < q.q1 - 1.5 * (q.q3 - q.q1)
                  OR p.puntos_totales > q.q3 + 1.5 * (q.q3 - q.q1) THEN 1 ELSE 0 END) AS atipicos_iqr
FROM nba_curated.partidos p
JOIN (
  SELECT temporada,
         percentile_approx(CAST(puntos_totales AS DOUBLE), 0.25) AS q1,
         percentile_approx(CAST(puntos_totales AS DOUBLE), 0.75) AS q3
  FROM nba_curated.partidos
  WHERE jugado = 1 AND game_type = 'Regular Season'
  GROUP BY temporada
) q ON q.temporada = p.temporada
WHERE p.jugado = 1 AND p.game_type = 'Regular Season' AND p.temporada >= 2000
GROUP BY p.temporada
ORDER BY p.temporada;

-- 10. Formato fisico: Parquet, Snappy, ubicacion en /nba/curated y particiones
DESCRIBE FORMATTED nba_curated.partidos;
DESCRIBE FORMATTED nba_curated.estadisticas_equipo;
DESCRIBE FORMATTED nba_curated.estadisticas_avanzadas;
SHOW PARTITIONS nba_curated.partidos;
