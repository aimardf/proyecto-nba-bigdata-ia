-- =============================================================================
-- 02_limpieza_curated.hql · HoopMind (Big Data + IA)
--
-- Limpia las tablas de /nba/dirty (y TeamStatisticsExtended de raw) y escribe la
-- zona /nba/curated en Parquet (Snappy), particionada por temporada.
--
--   nba_dirty.games, nba_dirty.team_statistics, nba_raw.team_statistics_ext
--        |  1. tipos, fechas y duplicados       -> nba_work.ts_tipado / games_tipado
--        |  2. rangos y reglas de baloncesto    -> nba_work.ts_reglas
--        |  3. imputacion con media movil       -> nba_work.ts_limpia
--        |  4. nombre oficial de cada equipo    -> nba_work.nombres_equipo
--        v  5. tablas finales                   -> nba_curated.*
--
-- Que resuelve cada paso (plan de ensuciado, informe v3.2 seccion 6):
--   duplicados   -> ROW_NUMBER() por (gameId, teamId) o por gameId
--   fechas       -> se reconoce el formato con RLIKE y se parsea con su patron
--   tipos        -> regexp_replace: fuera texto («112 pts») y coma decimal
--   imposibles   -> reglas de rango; % recalculado con anotados/intentados;
--                   puntos recalculados con la suma de los cuartos
--   nulos        -> primero reglas exactas (anotados = % x intentados,
--                   rebotes totales = ofensivos + defensivos); si no se puede,
--                   media de los 10 partidos ANTERIORES del equipo
--   nombres      -> nombre mas frecuente de cada teamId en cada temporada
--   local/visit. -> se contrasta con la marca home de TeamStatistics
--   marcador     -> se contrasta con los puntos (ya validados) de TeamStatistics
--
-- Columnas: las de la fuente pasan a snake_case en ingles (teamScore ->
-- team_score); las que creamos nosotros van en espanol (home_win, marca_*).
-- Requiere 01_tablas_entrada.hql (generado por el pipeline).
-- =============================================================================

SET hive.exec.dynamic.partition = true;
SET hive.exec.dynamic.partition.mode = nonstrict;
SET hive.exec.max.dynamic.partitions = 1000;
SET hive.exec.max.dynamic.partitions.pernode = 1000;
SET parquet.compression = SNAPPY;

CREATE DATABASE IF NOT EXISTS nba_work;
CREATE DATABASE IF NOT EXISTS nba_curated;

-- ---------------------------------------------------------------------------
-- Funciones de apoyo (macros de sesion)
-- ---------------------------------------------------------------------------
-- Texto -> numero: quita espacios y letras («112 pts» -> 112) y cambia coma por punto.
CREATE TEMPORARY MACRO num(s STRING)
  CAST(regexp_replace(regexp_replace(trim(s), ',', '.'), '[^0-9.-]', '') AS DOUBLE);

-- Texto vacio -> NULL
CREATE TEMPORARY MACRO txt(s STRING)
  CASE WHEN trim(s) = '' THEN NULL ELSE trim(s) END;

-- Valor dentro de [lo, hi] o NULL; y marca 1 si venia relleno pero fuera de rango
CREATE TEMPORARY MACRO rango(x DOUBLE, lo DOUBLE, hi DOUBLE)
  CASE WHEN x >= lo AND x <= hi THEN x ELSE NULL END;
CREATE TEMPORARY MACRO fuera(x DOUBLE, lo DOUBLE, hi DOUBLE)
  CASE WHEN x IS NOT NULL AND (x < lo OR x > hi) THEN 1 ELSE 0 END;

-- Fecha y hora en cualquiera de los formatos que aparecen
CREATE TEMPORARY MACRO fecha_hora(s STRING)
  CASE
    WHEN trim(s) RLIKE '^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}(\\.0+)?$'
      THEN CAST(substr(trim(s), 1, 19) AS TIMESTAMP)
    WHEN trim(s) RLIKE '^[0-9]{4}-[0-9]{2}-[0-9]{2}$'
      THEN CAST(concat(trim(s), ' 00:00:00') AS TIMESTAMP)
    WHEN trim(s) RLIKE '^[0-9]{2}/[0-9]{2}/[0-9]{4} [0-9]{2}:[0-9]{2}:[0-9]{2}$'
      THEN CAST(from_unixtime(unix_timestamp(trim(s), 'dd/MM/yyyy HH:mm:ss')) AS TIMESTAMP)
    WHEN trim(s) RLIKE '^[0-9]{4}/[0-9]{2}/[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}$'
      THEN CAST(from_unixtime(unix_timestamp(trim(s), 'yyyy/MM/dd HH:mm:ss')) AS TIMESTAMP)
    WHEN trim(s) RLIKE '^[0-9]{2}-[0-9]{2}-[0-9]{4} [0-9]{2}:[0-9]{2}:[0-9]{2}$'
      THEN CAST(from_unixtime(unix_timestamp(trim(s), 'MM-dd-yyyy HH:mm:ss')) AS TIMESTAMP)
    ELSE NULL
  END;

-- Temporada = anio en que empieza (la NBA empieza en octubre): 2025-26 -> 2025
CREATE TEMPORARY MACRO temporada_de(t TIMESTAMP)
  CASE WHEN month(t) >= 8 THEN year(t) ELSE year(t) - 1 END;


-- =============================================================================
-- 1. TeamStatistics: tipos, fechas y duplicados
-- =============================================================================
DROP TABLE IF EXISTS nba_work.ts_tipado;
CREATE TABLE nba_work.ts_tipado STORED AS PARQUET AS
SELECT * FROM (
  -- duplicados: una sola fila por (game_id, team_id); si hay copias, la que tenga fecha valida
  SELECT p.*,
         ROW_NUMBER() OVER (PARTITION BY game_id, team_id ORDER BY sin_fecha) AS rn
  FROM (
  SELECT
    CAST(num(gameId) AS BIGINT)          AS game_id,
    CAST(num(teamId) AS BIGINT)          AS team_id,
    CAST(num(opponentTeamId) AS BIGINT)  AS opponent_team_id,
    fecha_hora(gameDateTimeEst)          AS game_datetime,
    temporada_de(fecha_hora(gameDateTimeEst)) AS temporada,
    CASE WHEN fecha_hora(gameDateTimeEst) IS NULL THEN 1 ELSE 0 END AS sin_fecha,
    txt(teamCity)          AS team_city_txt,
    txt(teamName)          AS team_name_txt,
    txt(opponentTeamCity)  AS opponent_team_city_txt,
    txt(opponentTeamName)  AS opponent_team_name_txt,
    txt(gameType)          AS game_type,
    CAST(num(home) AS INT) AS home,
    CAST(num(win) AS INT)  AS win,
    num(teamScore)         AS team_score,
    num(opponentScore)     AS opponent_score,
    num(plusMinusPoints)   AS plus_minus_points,
    num(numMinutes)        AS num_minutes,
    num(assists)           AS assists,
    num(blocks)            AS blocks,
    num(steals)            AS steals,
    num(turnovers)         AS turnovers,
    num(foulsPersonal)     AS fouls_personal,
    num(fieldGoalsMade)            AS fgm,
    num(fieldGoalsAttempted)       AS fga,
    num(fieldGoalsPercentage)      AS fgp,
    num(threePointersMade)         AS tpm,
    num(threePointersAttempted)    AS tpa,
    num(threePointersPercentage)   AS tpp,
    num(freeThrowsMade)            AS ftm,
    num(freeThrowsAttempted)       AS fta,
    num(freeThrowsPercentage)      AS ftp,
    num(reboundsOffensive)  AS reb_off,
    num(reboundsDefensive)  AS reb_def,
    num(reboundsTotal)      AS reb_tot,
    num(q1Points) AS q1, num(q2Points) AS q2, num(q3Points) AS q3, num(q4Points) AS q4,
    num(ot1Points) AS ot1, num(ot2Points) AS ot2, num(otAllPoints) AS ot_all,
    num(benchPoints)          AS bench_points,
    num(biggestLead)          AS biggest_lead,
    num(biggestScoringRun)    AS biggest_scoring_run,
    num(leadChanges)          AS lead_changes,
    num(pointsFastBreak)      AS points_fast_break,
    num(pointsFromTurnovers)  AS points_from_turnovers,
    num(pointsInThePaint)     AS points_in_the_paint,
    num(pointsSecondChance)   AS points_second_chance,
    num(timesTied)            AS times_tied,
    num(timeoutsRemaining)    AS timeouts_remaining,
    num(reboundsTeam)         AS rebounds_team,
    num(turnoversTeam)        AS turnovers_team,
    num(seasonWins)           AS season_wins,
    num(seasonLosses)         AS season_losses
  FROM nba_dirty.team_statistics
) p
WHERE game_id IS NOT NULL AND team_id IS NOT NULL
) t
WHERE rn = 1;


-- =============================================================================
-- 2. TeamStatistics: rangos y reglas exactas
--    v = valores dentro de rango (lo imposible pasa a NULL)
--    r = valores recuperados con reglas que no inventan nada
-- =============================================================================
DROP TABLE IF EXISTS nba_work.ts_reglas;
CREATE TABLE nba_work.ts_reglas STORED AS PARQUET AS
SELECT
  r.*,
  -- puntos: el dato si es valido; si no, suma de cuartos; si no, 2*TC + T3 + TL
  CAST(CASE
    WHEN r.team_score_v IS NOT NULL THEN r.team_score_v
    WHEN r.suma_cuartos IS NOT NULL THEN r.suma_cuartos
    ELSE 2 * r.fgm_r + r.tpm_r + r.ftm_r
  END AS INT) AS team_score_final,
  CASE WHEN r.team_score_v IS NULL
        AND COALESCE(r.suma_cuartos, 2 * r.fgm_r + r.tpm_r + r.ftm_r) IS NOT NULL
       THEN 1 ELSE 0 END AS marca_score_recalculado,
  CASE WHEN r.team_score_v IS NOT NULL AND r.suma_cuartos IS NOT NULL
        AND r.team_score_v <> r.suma_cuartos
       THEN 1 ELSE 0 END AS marca_score_incoherente,
  -- porcentajes: el dato si es valido; si no, anotados / intentados
  COALESCE(r.fgp_v, CASE WHEN r.fga_r > 0 THEN round(r.fgm_r / r.fga_r, 3) END) AS fgp_r,
  COALESCE(r.tpp_v, CASE WHEN r.tpa_r > 0 THEN round(r.tpm_r / r.tpa_r, 3) END) AS tpp_r,
  COALESCE(r.ftp_v, CASE WHEN r.fta_r > 0 THEN round(r.ftm_r / r.fta_r, 3) END) AS ftp_r
FROM (
  SELECT
    v.*,
    -- anotados = % x intentados ; intentados = anotados / %
    COALESCE(v.fgm_v, round(v.fgp_v * v.fga_v)) AS fgm_r,
    COALESCE(v.fga_v, CASE WHEN v.fgp_v > 0 THEN round(v.fgm_v / v.fgp_v) END) AS fga_r,
    COALESCE(v.tpm_v, round(v.tpp_v * v.tpa_v)) AS tpm_r,
    COALESCE(v.tpa_v, CASE WHEN v.tpp_v > 0 THEN round(v.tpm_v / v.tpp_v) END) AS tpa_r,
    COALESCE(v.ftm_v, round(v.ftp_v * v.fta_v)) AS ftm_r,
    COALESCE(v.fta_v, CASE WHEN v.ftp_v > 0 THEN round(v.ftm_v / v.ftp_v) END) AS fta_r,
    -- rebotes totales = ofensivos + defensivos
    COALESCE(v.reb_tot_v, v.reb_off_v + v.reb_def_v) AS reb_tot_r,
    COALESCE(v.reb_off_v, v.reb_tot_v - v.reb_def_v) AS reb_off_r,
    COALESCE(v.reb_def_v, v.reb_tot_v - v.reb_off_v) AS reb_def_r,
    -- suma de cuartos (si los cuartos existen y no son todo ceros)
    CASE WHEN v.q1 IS NOT NULL AND v.q2 IS NOT NULL AND v.q3 IS NOT NULL AND v.q4 IS NOT NULL
          AND v.q1 + v.q2 + v.q3 + v.q4 > 0
         THEN v.q1 + v.q2 + v.q3 + v.q4 + COALESCE(v.ot_all, 0) END AS suma_cuartos,
    -- cuantos datos se han recuperado con estas reglas
      (CASE WHEN v.fgm_v IS NULL AND v.fgp_v IS NOT NULL AND v.fga_v IS NOT NULL THEN 1 ELSE 0 END)
    + (CASE WHEN v.fga_v IS NULL AND v.fgp_v > 0 AND v.fgm_v IS NOT NULL THEN 1 ELSE 0 END)
    + (CASE WHEN v.tpm_v IS NULL AND v.tpp_v IS NOT NULL AND v.tpa_v IS NOT NULL THEN 1 ELSE 0 END)
    + (CASE WHEN v.tpa_v IS NULL AND v.tpp_v > 0 AND v.tpm_v IS NOT NULL THEN 1 ELSE 0 END)
    + (CASE WHEN v.ftm_v IS NULL AND v.ftp_v IS NOT NULL AND v.fta_v IS NOT NULL THEN 1 ELSE 0 END)
    + (CASE WHEN v.fta_v IS NULL AND v.ftp_v > 0 AND v.ftm_v IS NOT NULL THEN 1 ELSE 0 END)
    + (CASE WHEN v.reb_tot_v IS NULL AND v.reb_off_v IS NOT NULL AND v.reb_def_v IS NOT NULL THEN 1 ELSE 0 END)
    + (CASE WHEN v.reb_off_v IS NULL AND v.reb_tot_v IS NOT NULL AND v.reb_def_v IS NOT NULL THEN 1 ELSE 0 END)
    + (CASE WHEN v.reb_def_v IS NULL AND v.reb_tot_v IS NOT NULL AND v.reb_off_v IS NOT NULL THEN 1 ELSE 0 END)
    + (CASE WHEN v.fgp_v IS NULL AND v.fgm_v IS NOT NULL AND v.fga_v > 0 THEN 1 ELSE 0 END)
    + (CASE WHEN v.tpp_v IS NULL AND v.tpm_v IS NOT NULL AND v.tpa_v > 0 THEN 1 ELSE 0 END)
    + (CASE WHEN v.ftp_v IS NULL AND v.ftm_v IS NOT NULL AND v.fta_v > 0 THEN 1 ELSE 0 END)
      AS n_recuperados_regla
  FROM (
    SELECT
      game_id, team_id, opponent_team_id, game_datetime, temporada,
      team_city_txt, team_name_txt, opponent_team_city_txt, opponent_team_name_txt,
      game_type, home, win,
      rango(team_score, 0, 200)        AS team_score_v,
      rango(opponent_score, 0, 200)    AS opponent_score_v,
      rango(plus_minus_points, -100, 100) AS plus_minus_v,
      rango(num_minutes, 0, 500)       AS num_minutes_v,
      rango(assists, 0, 200)  AS assists_v,
      rango(blocks, 0, 200)   AS blocks_v,
      rango(steals, 0, 200)   AS steals_v,
      rango(turnovers, 0, 200) AS turnovers_v,
      rango(fouls_personal, 0, 200) AS fouls_personal_v,
      rango(fgm, 0, 200) AS fgm_v, rango(fga, 0, 200) AS fga_v, rango(fgp, 0, 1) AS fgp_v,
      rango(tpm, 0, 200) AS tpm_v, rango(tpa, 0, 200) AS tpa_v, rango(tpp, 0, 1) AS tpp_v,
      rango(ftm, 0, 200) AS ftm_v, rango(fta, 0, 200) AS fta_v, rango(ftp, 0, 1) AS ftp_v,
      rango(reb_off, 0, 200) AS reb_off_v,
      rango(reb_def, 0, 200) AS reb_def_v,
      rango(reb_tot, 0, 200) AS reb_tot_v,
      rango(q1, 0, 100) AS q1, rango(q2, 0, 100) AS q2,
      rango(q3, 0, 100) AS q3, rango(q4, 0, 100) AS q4,
      COALESCE(rango(ot1, 0, 100), 0)    AS ot1,      -- vacio = no hubo prorroga = 0
      COALESCE(rango(ot2, 0, 100), 0)    AS ot2,
      COALESCE(rango(ot_all, 0, 200), 0) AS ot_all,
      bench_points, biggest_lead, biggest_scoring_run, lead_changes, points_fast_break,
      points_from_turnovers, points_in_the_paint, points_second_chance, times_tied,
      timeouts_remaining, rebounds_team, turnovers_team, season_wins, season_losses,
      fuera(team_score, 0, 200) + fuera(opponent_score, 0, 200)
        + fuera(fgp, 0, 1) + fuera(tpp, 0, 1) + fuera(ftp, 0, 1)
        + fuera(assists, 0, 200) + fuera(reb_tot, 0, 200) + fuera(turnovers, 0, 200)
        AS marca_fuera_rango
    FROM nba_work.ts_tipado
  ) v
) r;


-- =============================================================================
-- 3. TeamStatistics: imputacion de lo que no se puede deducir
--    Media de los 10 partidos ANTERIORES del mismo equipo (nunca el propio
--    partido ni partidos futuros). Si no hay, media de TODOS sus partidos
--    anteriores. Nunca la media de la temporada entera (incluiria partidos
--    futuros: fuga de datos). No usamos la media global: en temporadas antiguas hay estadisticas que no
--    existian (triples antes de 1979) y no queremos inventarlas.
-- =============================================================================
DROP TABLE IF EXISTS nba_work.ts_limpia;
CREATE TABLE nba_work.ts_limpia STORED AS PARQUET AS
SELECT
  game_id, team_id, opponent_team_id, game_datetime, temporada,
  team_city_txt, team_name_txt, opponent_team_city_txt, opponent_team_name_txt,
  game_type, home, win,
  team_score_final AS team_score,
  CAST(opponent_score_v AS INT) AS opponent_score,
  CAST(plus_minus_v AS INT)     AS plus_minus_points,
  num_minutes_v                 AS num_minutes,
  CAST(COALESCE(assists_v, round(m10_assists), round(mh_assists)) AS INT) AS assists,
  CAST(COALESCE(blocks_v, round(m10_blocks), round(mh_blocks)) AS INT) AS blocks,
  CAST(COALESCE(steals_v, round(m10_steals), round(mh_steals)) AS INT) AS steals,
  CAST(COALESCE(turnovers_v, round(m10_turnovers), round(mh_turnovers)) AS INT) AS turnovers,
  CAST(COALESCE(fouls_personal_v, round(m10_fouls_personal), round(mh_fouls_personal)) AS INT) AS fouls_personal,
  CAST(COALESCE(fgm_r, round(m10_field_goals_made), round(mh_field_goals_made)) AS INT) AS field_goals_made,
  CAST(COALESCE(fga_r, round(m10_field_goals_attempted), round(mh_field_goals_attempted)) AS INT) AS field_goals_attempted,
  COALESCE(fgp_r, round(m10_field_goals_percentage, 3), round(mh_field_goals_percentage, 3)) AS field_goals_percentage,
  CAST(COALESCE(tpm_r, round(m10_three_pointers_made), round(mh_three_pointers_made)) AS INT) AS three_pointers_made,
  CAST(COALESCE(tpa_r, round(m10_three_pointers_attempted), round(mh_three_pointers_attempted)) AS INT) AS three_pointers_attempted,
  COALESCE(tpp_r, round(m10_three_pointers_percentage, 3), round(mh_three_pointers_percentage, 3)) AS three_pointers_percentage,
  CAST(COALESCE(ftm_r, round(m10_free_throws_made), round(mh_free_throws_made)) AS INT) AS free_throws_made,
  CAST(COALESCE(fta_r, round(m10_free_throws_attempted), round(mh_free_throws_attempted)) AS INT) AS free_throws_attempted,
  COALESCE(ftp_r, round(m10_free_throws_percentage, 3), round(mh_free_throws_percentage, 3)) AS free_throws_percentage,
  CAST(COALESCE(reb_off_r, round(m10_rebounds_offensive), round(mh_rebounds_offensive)) AS INT) AS rebounds_offensive,
  CAST(COALESCE(reb_def_r, round(m10_rebounds_defensive), round(mh_rebounds_defensive)) AS INT) AS rebounds_defensive,
  CAST(COALESCE(reb_tot_r, round(m10_rebounds_total), round(mh_rebounds_total)) AS INT) AS rebounds_total,
  CAST(q1 AS INT) AS q1_points, CAST(q2 AS INT) AS q2_points,
  CAST(q3 AS INT) AS q3_points, CAST(q4 AS INT) AS q4_points,
  CAST(ot1 AS INT) AS ot1_points, CAST(ot2 AS INT) AS ot2_points, CAST(ot_all AS INT) AS ot_all_points,
  CASE WHEN ot_all > 0 THEN 1 ELSE 0 END AS prorroga,
  CAST(bench_points AS INT)          AS bench_points,
  CAST(biggest_lead AS INT)          AS biggest_lead,
  CAST(biggest_scoring_run AS INT)   AS biggest_scoring_run,
  CAST(lead_changes AS INT)          AS lead_changes,
  CAST(points_fast_break AS INT)     AS points_fast_break,
  CAST(points_from_turnovers AS INT) AS points_from_turnovers,
  CAST(points_in_the_paint AS INT)   AS points_in_the_paint,
  CAST(points_second_chance AS INT)  AS points_second_chance,
  CAST(times_tied AS INT)            AS times_tied,
  CAST(timeouts_remaining AS INT)    AS timeouts_remaining,
  CAST(rebounds_team AS INT)         AS rebounds_team,
  CAST(turnovers_team AS INT)        AS turnovers_team,
  CAST(season_wins AS INT)           AS season_wins,
  CAST(season_losses AS INT)         AS season_losses,
  marca_fuera_rango, marca_score_recalculado, marca_score_incoherente, n_recuperados_regla,
    (CASE WHEN assists_v IS NULL THEN 1 ELSE 0 END)
  + (CASE WHEN blocks_v IS NULL THEN 1 ELSE 0 END)
  + (CASE WHEN steals_v IS NULL THEN 1 ELSE 0 END)
  + (CASE WHEN turnovers_v IS NULL THEN 1 ELSE 0 END)
  + (CASE WHEN fouls_personal_v IS NULL THEN 1 ELSE 0 END)
  + (CASE WHEN fgm_r IS NULL THEN 1 ELSE 0 END)
  + (CASE WHEN fga_r IS NULL THEN 1 ELSE 0 END)
  + (CASE WHEN fgp_r IS NULL THEN 1 ELSE 0 END)
  + (CASE WHEN tpm_r IS NULL THEN 1 ELSE 0 END)
  + (CASE WHEN tpa_r IS NULL THEN 1 ELSE 0 END)
  + (CASE WHEN tpp_r IS NULL THEN 1 ELSE 0 END)
  + (CASE WHEN ftm_r IS NULL THEN 1 ELSE 0 END)
  + (CASE WHEN fta_r IS NULL THEN 1 ELSE 0 END)
  + (CASE WHEN ftp_r IS NULL THEN 1 ELSE 0 END)
  + (CASE WHEN reb_off_r IS NULL THEN 1 ELSE 0 END)
  + (CASE WHEN reb_def_r IS NULL THEN 1 ELSE 0 END)
  + (CASE WHEN reb_tot_r IS NULL THEN 1 ELSE 0 END)
    AS n_huecos_antes_de_imputar
FROM (
  -- medias de los partidos ANTERIORES de cada equipo (m10_ = ultimos 10, mh_ = todos)
  SELECT
    r.*,
    AVG(assists_v) OVER w10 AS m10_assists,
    AVG(assists_v) OVER wt  AS mh_assists,
    AVG(blocks_v) OVER w10 AS m10_blocks,
    AVG(blocks_v) OVER wt  AS mh_blocks,
    AVG(steals_v) OVER w10 AS m10_steals,
    AVG(steals_v) OVER wt  AS mh_steals,
    AVG(turnovers_v) OVER w10 AS m10_turnovers,
    AVG(turnovers_v) OVER wt  AS mh_turnovers,
    AVG(fouls_personal_v) OVER w10 AS m10_fouls_personal,
    AVG(fouls_personal_v) OVER wt  AS mh_fouls_personal,
    AVG(fgm_r) OVER w10 AS m10_field_goals_made,
    AVG(fgm_r) OVER wt  AS mh_field_goals_made,
    AVG(fga_r) OVER w10 AS m10_field_goals_attempted,
    AVG(fga_r) OVER wt  AS mh_field_goals_attempted,
    AVG(fgp_r) OVER w10 AS m10_field_goals_percentage,
    AVG(fgp_r) OVER wt  AS mh_field_goals_percentage,
    AVG(tpm_r) OVER w10 AS m10_three_pointers_made,
    AVG(tpm_r) OVER wt  AS mh_three_pointers_made,
    AVG(tpa_r) OVER w10 AS m10_three_pointers_attempted,
    AVG(tpa_r) OVER wt  AS mh_three_pointers_attempted,
    AVG(tpp_r) OVER w10 AS m10_three_pointers_percentage,
    AVG(tpp_r) OVER wt  AS mh_three_pointers_percentage,
    AVG(ftm_r) OVER w10 AS m10_free_throws_made,
    AVG(ftm_r) OVER wt  AS mh_free_throws_made,
    AVG(fta_r) OVER w10 AS m10_free_throws_attempted,
    AVG(fta_r) OVER wt  AS mh_free_throws_attempted,
    AVG(ftp_r) OVER w10 AS m10_free_throws_percentage,
    AVG(ftp_r) OVER wt  AS mh_free_throws_percentage,
    AVG(reb_off_r) OVER w10 AS m10_rebounds_offensive,
    AVG(reb_off_r) OVER wt  AS mh_rebounds_offensive,
    AVG(reb_def_r) OVER w10 AS m10_rebounds_defensive,
    AVG(reb_def_r) OVER wt  AS mh_rebounds_defensive,
    AVG(reb_tot_r) OVER w10 AS m10_rebounds_total,
    AVG(reb_tot_r) OVER wt  AS mh_rebounds_total
  FROM nba_work.ts_reglas r
  WINDOW
    w10 AS (PARTITION BY team_id ORDER BY game_datetime, game_id ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING),
    wt  AS (PARTITION BY team_id ORDER BY game_datetime, game_id ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
) m;


-- =============================================================================
-- 4. Games: tipos, fechas y duplicados
-- =============================================================================
DROP TABLE IF EXISTS nba_work.games_tipado;
CREATE TABLE nba_work.games_tipado STORED AS PARQUET AS
SELECT * FROM (
  -- duplicados: una sola fila por game_id
  SELECT p.*, ROW_NUMBER() OVER (PARTITION BY game_id ORDER BY sin_fecha) AS rn
  FROM (
  SELECT
    CAST(num(gameId) AS BIGINT)     AS game_id,
    fecha_hora(gameDateTimeEst)     AS game_datetime,
    temporada_de(fecha_hora(gameDateTimeEst)) AS temporada,
    CASE WHEN fecha_hora(gameDateTimeEst) IS NULL THEN 1 ELSE 0 END AS sin_fecha,
    txt(hometeamCity)               AS home_team_city_txt,
    txt(hometeamName)               AS home_team_name_txt,
    CAST(num(hometeamId) AS BIGINT) AS home_team_id,
    txt(awayteamCity)               AS away_team_city_txt,
    txt(awayteamName)               AS away_team_name_txt,
    CAST(num(awayteamId) AS BIGINT) AS away_team_id,
    rango(num(homeScore), 0, 200)   AS home_score_v,
    rango(num(awayScore), 0, 200)   AS away_score_v,
    CAST(num(winner) AS BIGINT)     AS winner,
    txt(gameType)                   AS game_type,
    txt(gameSubtype)                AS game_subtype,
    CAST(num(seriesGameNumber) AS INT) AS series_game_number,
    CAST(num(attendance) AS INT)    AS attendance,
    CAST(num(arenaId) AS BIGINT)    AS arena_id,
    txt(arenaName)                  AS arena_name,
    txt(arenaCity)                  AS arena_city,
    txt(arenaState)                 AS arena_state
  FROM nba_dirty.games
  ) p
  WHERE game_id IS NOT NULL
) t
WHERE rn = 1;


-- =============================================================================
-- 5. Nombre oficial de cada equipo en cada temporada (el mas frecuente)
--    Corrige MAYUSCULAS, minusculas, espacios y erratas sin tabla hecha a mano.
-- =============================================================================
DROP TABLE IF EXISTS nba_work.nombres_equipo;
CREATE TABLE nba_work.nombres_equipo STORED AS PARQUET AS
SELECT team_id, temporada, team_city, team_name, n_filas
FROM (
  SELECT team_id, temporada, team_city, team_name, n_filas,
         ROW_NUMBER() OVER (PARTITION BY team_id, temporada ORDER BY n_filas DESC, team_name) AS rn
  FROM (
    SELECT team_id, temporada, team_city, team_name, COUNT(*) AS n_filas
    FROM (
      SELECT team_id, temporada, team_city_txt AS team_city, team_name_txt AS team_name
        FROM nba_work.ts_tipado
      UNION ALL
      SELECT home_team_id, temporada, home_team_city_txt, home_team_name_txt
        FROM nba_work.games_tipado
      UNION ALL
      SELECT away_team_id, temporada, away_team_city_txt, away_team_name_txt
        FROM nba_work.games_tipado
    ) u
    WHERE team_id IS NOT NULL AND temporada IS NOT NULL AND team_name IS NOT NULL
    GROUP BY team_id, temporada, team_city, team_name
  ) c
) r
WHERE rn = 1;


-- =============================================================================
-- 6. Tablas finales en /nba/curated (Parquet + Snappy, particion por temporada)
-- =============================================================================
DROP TABLE IF EXISTS nba_curated.estadisticas_equipo;
CREATE EXTERNAL TABLE nba_curated.estadisticas_equipo (
  game_id BIGINT, team_id BIGINT, opponent_team_id BIGINT,
  game_datetime TIMESTAMP, game_date DATE, game_type STRING,
  team_city STRING, team_name STRING, opponent_team_city STRING, opponent_team_name STRING,
  home INT, win INT, team_score INT, opponent_score INT, plus_minus_points INT, num_minutes DOUBLE,
  assists INT, blocks INT, steals INT, turnovers INT, fouls_personal INT,
  field_goals_made INT, field_goals_attempted INT, field_goals_percentage DOUBLE,
  three_pointers_made INT, three_pointers_attempted INT, three_pointers_percentage DOUBLE,
  free_throws_made INT, free_throws_attempted INT, free_throws_percentage DOUBLE,
  rebounds_offensive INT, rebounds_defensive INT, rebounds_total INT,
  q1_points INT, q2_points INT, q3_points INT, q4_points INT,
  ot1_points INT, ot2_points INT, ot_all_points INT, prorroga INT,
  bench_points INT, biggest_lead INT, biggest_scoring_run INT, lead_changes INT,
  points_fast_break INT, points_from_turnovers INT, points_in_the_paint INT,
  points_second_chance INT, times_tied INT, timeouts_remaining INT,
  rebounds_team INT, turnovers_team INT, season_wins INT, season_losses INT,
  marca_fuera_rango INT, marca_score_recalculado INT, marca_score_incoherente INT,
  n_recuperados_regla INT, n_huecos_antes_de_imputar INT
)
PARTITIONED BY (temporada INT)
STORED AS PARQUET
LOCATION '/nba/curated/estadisticas_equipo'
TBLPROPERTIES ("parquet.compression" = "SNAPPY");

INSERT OVERWRITE TABLE nba_curated.estadisticas_equipo PARTITION (temporada)
SELECT
  t.game_id, t.team_id, t.opponent_team_id,
  t.game_datetime, CAST(t.game_datetime AS DATE), t.game_type,
  COALESCE(eq.team_city, t.team_city_txt), COALESCE(eq.team_name, t.team_name_txt),
  COALESCE(rv.team_city, t.opponent_team_city_txt), COALESCE(rv.team_name, t.opponent_team_name_txt),
  t.home, t.win, t.team_score, t.opponent_score, t.plus_minus_points, t.num_minutes,
  t.assists, t.blocks, t.steals, t.turnovers, t.fouls_personal,
  t.field_goals_made, t.field_goals_attempted, t.field_goals_percentage,
  t.three_pointers_made, t.three_pointers_attempted, t.three_pointers_percentage,
  t.free_throws_made, t.free_throws_attempted, t.free_throws_percentage,
  t.rebounds_offensive, t.rebounds_defensive, t.rebounds_total,
  t.q1_points, t.q2_points, t.q3_points, t.q4_points,
  t.ot1_points, t.ot2_points, t.ot_all_points, t.prorroga,
  t.bench_points, t.biggest_lead, t.biggest_scoring_run, t.lead_changes,
  t.points_fast_break, t.points_from_turnovers, t.points_in_the_paint,
  t.points_second_chance, t.times_tied, t.timeouts_remaining,
  t.rebounds_team, t.turnovers_team, t.season_wins, t.season_losses,
  t.marca_fuera_rango, t.marca_score_recalculado, t.marca_score_incoherente,
  t.n_recuperados_regla, t.n_huecos_antes_de_imputar,
  t.temporada
FROM nba_work.ts_limpia t
LEFT JOIN nba_work.nombres_equipo eq ON eq.team_id = t.team_id AND eq.temporada = t.temporada
LEFT JOIN nba_work.nombres_equipo rv ON rv.team_id = t.opponent_team_id AND rv.temporada = t.temporada
WHERE t.temporada IS NOT NULL;


DROP TABLE IF EXISTS nba_curated.partidos;
CREATE EXTERNAL TABLE nba_curated.partidos (
  game_id BIGINT, game_datetime TIMESTAMP, game_date DATE,
  game_type STRING, game_subtype STRING, series_game_number INT,
  home_team_id BIGINT, home_team_city STRING, home_team_name STRING,
  away_team_id BIGINT, away_team_city STRING, away_team_name STRING,
  home_score INT, away_score INT, winner BIGINT,
  attendance INT, arena_id BIGINT, arena_name STRING, arena_city STRING, arena_state STRING,
  home_win INT, puntos_totales INT, jugado INT, prorroga INT,
  marca_intercambio INT, marca_marcador INT, marca_ganador_incoherente INT
)
PARTITIONED BY (temporada INT)
STORED AS PARQUET
LOCATION '/nba/curated/partidos'
TBLPROPERTIES ("parquet.compression" = "SNAPPY");

INSERT OVERWRITE TABLE nba_curated.partidos PARTITION (temporada)
SELECT
  s.game_id, s.game_datetime, CAST(s.game_datetime AS DATE),
  s.game_type, s.game_subtype, s.series_game_number,
  s.h_id, COALESCE(loc.team_city, s.h_city_txt), COALESCE(loc.team_name, s.h_name_txt),
  s.a_id, COALESCE(vis.team_city, s.a_city_txt), COALESCE(vis.team_name, s.a_name_txt),
  CAST(s.home_score AS INT), CAST(s.away_score AS INT), s.winner,
  s.attendance, s.arena_id, s.arena_name, s.arena_city, s.arena_state,
  -- salida del objetivo 1: 1 si gana el local (como en el notebook: winner = local)
  CASE WHEN s.jugado = 1 THEN
    CASE WHEN s.winner = s.h_id THEN 1
         WHEN s.winner = s.a_id THEN 0
         WHEN s.home_score > s.away_score THEN 1 ELSE 0 END
  END AS home_win,
  -- salida del objetivo 2: puntos totales del partido (incluye prorrogas)
  CASE WHEN s.jugado = 1 THEN CAST(s.home_score + s.away_score AS INT) END AS puntos_totales,
  s.jugado, s.prorroga,
  s.marca_intercambio, s.marca_marcador,
  CASE WHEN s.jugado = 1 AND s.winner IS NOT NULL
        AND s.winner <> (CASE WHEN s.home_score > s.away_score THEN s.h_id ELSE s.a_id END)
       THEN 1 ELSE 0 END AS marca_ganador_incoherente,
  s.temporada
FROM (
  SELECT
    x.*,
    CASE WHEN x.marca_intercambio = 1 THEN x.away_team_id ELSE x.home_team_id END AS h_id,
    CASE WHEN x.marca_intercambio = 1 THEN x.home_team_id ELSE x.away_team_id END AS a_id,
    CASE WHEN x.marca_intercambio = 1 THEN x.away_team_city_txt ELSE x.home_team_city_txt END AS h_city_txt,
    CASE WHEN x.marca_intercambio = 1 THEN x.home_team_city_txt ELSE x.away_team_city_txt END AS a_city_txt,
    CASE WHEN x.marca_intercambio = 1 THEN x.away_team_name_txt ELSE x.home_team_name_txt END AS h_name_txt,
    CASE WHEN x.marca_intercambio = 1 THEN x.home_team_name_txt ELSE x.away_team_name_txt END AS a_name_txt,
    -- marcador: manda TeamStatistics (puntos ya validados con los cuartos)
    COALESCE(x.ts_local_pts,     CASE WHEN x.marca_intercambio = 1 THEN x.away_score_v ELSE x.home_score_v END) AS home_score,
    COALESCE(x.ts_visitante_pts, CASE WHEN x.marca_intercambio = 1 THEN x.home_score_v ELSE x.away_score_v END) AS away_score,
    CASE
      WHEN x.ts_local_pts IS NOT NULL
       AND (CASE WHEN x.marca_intercambio = 1 THEN x.away_score_v ELSE x.home_score_v END) IS NULL THEN 1
      WHEN x.ts_local_pts IS NOT NULL
       AND (CASE WHEN x.marca_intercambio = 1 THEN x.away_score_v ELSE x.home_score_v END) <> x.ts_local_pts THEN 1
      WHEN x.ts_visitante_pts IS NOT NULL
       AND (CASE WHEN x.marca_intercambio = 1 THEN x.home_score_v ELSE x.away_score_v END) IS NULL THEN 1
      WHEN x.ts_visitante_pts IS NOT NULL
       AND (CASE WHEN x.marca_intercambio = 1 THEN x.home_score_v ELSE x.away_score_v END) <> x.ts_visitante_pts THEN 1
      ELSE 0
    END AS marca_marcador,
    CASE WHEN COALESCE(x.ts_local_pts, CASE WHEN x.marca_intercambio = 1 THEN x.away_score_v ELSE x.home_score_v END) > 0
          AND COALESCE(x.ts_visitante_pts, CASE WHEN x.marca_intercambio = 1 THEN x.home_score_v ELSE x.away_score_v END) > 0
         THEN 1 ELSE 0 END AS jugado
  FROM (
    SELECT
      g.*,
      th.team_score AS ts_local_pts,
      ta.team_score AS ts_visitante_pts,
      th.prorroga AS prorroga,
      -- Games dice local = X, pero TeamStatistics dice que X jugo fuera (y al reves)
      CASE WHEN th.team_id IS NOT NULL AND ta.team_id IS NOT NULL
            AND g.home_team_id = ta.team_id AND g.away_team_id = th.team_id
           THEN 1 ELSE 0 END AS marca_intercambio
    FROM nba_work.games_tipado g
    LEFT JOIN (SELECT game_id, team_id, team_score, prorroga FROM nba_work.ts_limpia WHERE home = 1) th
      ON th.game_id = g.game_id
    LEFT JOIN (SELECT game_id, team_id, team_score FROM nba_work.ts_limpia WHERE home = 0) ta
      ON ta.game_id = g.game_id
  ) x
) s
LEFT JOIN nba_work.nombres_equipo loc ON loc.team_id = s.h_id AND loc.temporada = s.temporada
LEFT JOIN nba_work.nombres_equipo vis ON vis.team_id = s.a_id AND vis.temporada = s.temporada
WHERE s.temporada IS NOT NULL;


-- =============================================================================
-- 7. TeamStatisticsExtended: no se ensucia; se tipa, se quitan duplicados y se
--    guarda solo lo que usaremos (ratings, ritmo y estilo). Va directo de raw.
--    Fuera: seasonWins/seasonLosses (posible fuga) y lo que repite TeamStatistics.
-- =============================================================================
DROP TABLE IF EXISTS nba_curated.estadisticas_avanzadas;
CREATE EXTERNAL TABLE nba_curated.estadisticas_avanzadas (
  game_id BIGINT, team_id BIGINT, game_datetime TIMESTAMP, home INT,
  offensive_rating DOUBLE, defensive_rating DOUBLE, net_rating DOUBLE,
  pace DOUBLE, possessions DOUBLE,
  effective_field_goal_percentage DOUBLE, true_shooting_percentage DOUBLE,
  assist_percentage DOUBLE, assist_to_turnover_ratio DOUBLE,
  offensive_rebound_percentage DOUBLE, defensive_rebound_percentage DOUBLE,
  rebound_percentage DOUBLE, team_turnover_percentage DOUBLE,
  free_throw_attempt_rate DOUBLE,
  percent_field_goal_attempts_3point DOUBLE,
  percent_points_3point DOUBLE, percent_points_in_paint DOUBLE,
  percent_points_fast_break DOUBLE, percent_points_free_throw DOUBLE
)
PARTITIONED BY (temporada INT)
STORED AS PARQUET
LOCATION '/nba/curated/estadisticas_avanzadas'
TBLPROPERTIES ("parquet.compression" = "SNAPPY");

INSERT OVERWRITE TABLE nba_curated.estadisticas_avanzadas PARTITION (temporada)
SELECT game_id, team_id, game_datetime, home,
       offensive_rating, defensive_rating, net_rating, pace, possessions,
       effective_field_goal_percentage, true_shooting_percentage,
       assist_percentage, assist_to_turnover_ratio,
       offensive_rebound_percentage, defensive_rebound_percentage,
       rebound_percentage, team_turnover_percentage, free_throw_attempt_rate,
       percent_field_goal_attempts_3point, percent_points_3point, percent_points_in_paint,
       percent_points_fast_break, percent_points_free_throw,
       temporada
FROM (
  SELECT p.*, ROW_NUMBER() OVER (PARTITION BY game_id, team_id ORDER BY sin_fecha) AS rn
  FROM (
  SELECT
    CAST(num(gameId) AS BIGINT) AS game_id,
    CAST(num(teamId) AS BIGINT) AS team_id,
    fecha_hora(gameDateTimeEst) AS game_datetime,
    temporada_de(fecha_hora(gameDateTimeEst)) AS temporada,
    CASE WHEN fecha_hora(gameDateTimeEst) IS NULL THEN 1 ELSE 0 END AS sin_fecha,
    CAST(num(home) AS INT) AS home,
    num(offensiveRating)  AS offensive_rating,
    num(defensiveRating)  AS defensive_rating,
    num(netRating)        AS net_rating,
    num(pace)             AS pace,
    num(possessions)      AS possessions,
    num(effectiveFieldGoalPercentage) AS effective_field_goal_percentage,
    num(trueShootingPercentage)       AS true_shooting_percentage,
    num(assistPercentage)             AS assist_percentage,
    num(assistToTurnoverRatio)        AS assist_to_turnover_ratio,
    num(offensiveReboundPercentage)   AS offensive_rebound_percentage,
    num(defensiveReboundPercentage)   AS defensive_rebound_percentage,
    num(reboundPercentage)            AS rebound_percentage,
    num(teamTurnoverPercentage)       AS team_turnover_percentage,
    num(freeThrowAttemptRate)         AS free_throw_attempt_rate,
    num(percentFieldGoalAttempts3Point) AS percent_field_goal_attempts_3point,
    num(percentPoints3Point)          AS percent_points_3point,
    num(percentPointsInPaint)         AS percent_points_in_paint,
    num(percentPointsFastBreak)       AS percent_points_fast_break,
    num(percentPointsFreeThrow)       AS percent_points_free_throw
  FROM nba_raw.team_statistics_ext
  ) p
  WHERE game_id IS NOT NULL AND team_id IS NOT NULL
) e
WHERE rn = 1 AND temporada IS NOT NULL;
