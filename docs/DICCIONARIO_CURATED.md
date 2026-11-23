# Diccionario de datos · zona curated (HoopMind)

Versión 2 · lunes 23/11/2026 (v1: jueves 19/11) · generado con `hive/02_limpieza_curated.hql`.

## Zonas de HDFS

| Zona | Ruta | Formato | Qué hay |
|---|---|---|---|
| raw | `/nba/raw/fecha=2026-11-17/` | CSV | Ficheros originales de Kaggle (eoinamoore, CC0). No se tocan. |
| dirty | `/nba/dirty/games/`, `/nba/dirty/team_statistics/` | CSV | Salida de `scripts/ensuciar.py --seed 42`. Sin `officials` ni `coachId`. |
| dirty (log) | `/nba/dirty/log/` | CSV + JSON | `ensuciado_log.csv` (una fila por celda cambiada) y `ensuciado_resumen.json`. |
| staging | `/nba/staging/team_statistics_ext/` | CSV | Copia de `TeamStatisticsExtended.csv` en su propia carpeta para que Hive la lea como tabla. |
| curated | `/nba/curated/<tabla>/temporada=AAAA/` | Parquet (Snappy) | Datos limpios y tipados. Es la entrada de la parte de IA. |

Bases de datos de Hive: `nba_raw` y `nba_dirty` (tablas externas sobre CSV, todo STRING), `nba_work` (pasos intermedios, Parquet) y `nba_curated` (tablas finales).

## Convenciones

- Las columnas que vienen de la fuente pasan a **snake_case en inglés**: `teamScore` → `team_score`, `hometeamId` → `home_team_id`.
- Las columnas que **creamos nosotros** van en español: `home_win`, `puntos_totales`, `jugado`, `prorroga` y todas las `marca_*`.
- `temporada` es el año en que empieza la temporada (2025-26 → 2025). Se calcula a partir de la fecha: si el mes es agosto o posterior, ese año; si no, el año anterior. Todas las tablas están particionadas por esta columna.
- Los porcentajes van de 0 a 1 (0,451 = 45,1 %).
- Los identificadores de equipo son los de NBA.com (`1610612737` = Hawks…).

---

## `nba_curated.partidos`

Una fila por partido (sale de `Games.csv`). Es la tabla principal de los dos objetivos.

| Columna | Tipo | Descripción | Origen / regla |
|---|---|---|---|
| game_id | BIGINT | Id del partido. Clave. Los dígitos 2-3 son la temporada (225… = 2025-26). | gameId |
| game_datetime | TIMESTAMP | Fecha y hora (hora del Este de EE. UU.) | gameDateTimeEst, leído en 5 formatos |
| game_date | DATE | Solo la fecha | de game_datetime |
| game_type | STRING | Regular Season, Playoffs, Preseason, Play-in Tournament, NBA Cup… | gameType |
| game_subtype | STRING | Subtipo (casi siempre vacío) | gameSubtype |
| series_game_number | INT | Número de partido en la serie de playoffs | seriesGameNumber |
| home_team_id | BIGINT | Equipo local | hometeamId, contrastado con TeamStatistics |
| home_team_city, home_team_name | STRING | Ciudad y nombre del local | nombre oficial más frecuente del equipo en la temporada |
| away_team_id | BIGINT | Equipo visitante | awayteamId, contrastado con TeamStatistics |
| away_team_city, away_team_name | STRING | Ciudad y nombre del visitante | ídem |
| home_score, away_score | INT | Puntos del local y del visitante | TeamStatistics (validados con la suma de cuartos); si no hay, Games |
| winner | BIGINT | Id del equipo ganador (no es 0/1) | winner |
| attendance | INT | Público. **No es entrada del modelo** (no se conoce antes del partido) | attendance |
| arena_id, arena_name, arena_city, arena_state | BIGINT / STRING | Pabellón | arena* |
| **home_win** | INT | **Salida del objetivo 1.** 1 si gana el local, 0 si no. NULL si no se ha jugado | winner = home_team_id |
| **puntos_totales** | INT | **Salida del objetivo 2.** home_score + away_score (incluye prórrogas). NULL si no se ha jugado | |
| jugado | INT | 1 si los dos marcadores son mayores que 0. Los partidos futuros quedan con 0 | |
| prorroga | INT | 1 si hubo prórroga (otAllPoints del local > 0). NULL si no hay fila de equipo | |
| marca_intercambio | INT | 1 si Games tenía local y visitante al revés y se ha corregido | contraste con `home` de TeamStatistics |
| marca_marcador | INT | 1 si el marcador de Games no coincidía con TeamStatistics y se ha corregido | |
| marca_ganador_incoherente | INT | 1 si winner no coincide con el que más puntos tiene (no debería pasar) | |
| temporada | INT | Partición | de game_datetime |

**Fuera de curated**: `officials` (nombres de árbitros, minimización RGPD), `gameLabel` y `gameSubLabel` (casi vacías) y `gameDate` (repite la fecha).

## `nba_curated.estadisticas_equipo`

Dos filas por partido, una por equipo (sale de `TeamStatistics.csv`). De aquí salen las **entradas** del modelo, siempre como medias de partidos **anteriores**.

| Columna | Tipo | Descripción |
|---|---|---|
| game_id, team_id | BIGINT | Clave (partido, equipo) |
| opponent_team_id | BIGINT | Rival |
| game_datetime, game_date, game_type | | Como en partidos |
| team_city, team_name, opponent_team_city, opponent_team_name | STRING | Nombres oficiales (más frecuente por equipo y temporada) |
| home | INT | 1 si el equipo juega en casa |
| win | INT | 1 si gana. **Es resultado: no es entrada** |
| team_score, opponent_score, plus_minus_points | INT | Puntos a favor, en contra y diferencia. **Resultado del propio partido** |
| num_minutes | DOUBLE | Minutos del equipo (240 sin prórroga) |
| assists, blocks, steals, turnovers, fouls_personal | INT | Asistencias, tapones, robos, pérdidas y faltas |
| field_goals_made / _attempted / _percentage | INT / INT / DOUBLE | Tiros de campo anotados, intentados y % (0-1) |
| three_pointers_made / _attempted / _percentage | | Triples |
| free_throws_made / _attempted / _percentage | | Tiros libres |
| rebounds_offensive, rebounds_defensive, rebounds_total | INT | Rebotes |
| q1_points … q4_points | INT | Puntos por cuarto |
| ot1_points, ot2_points, ot_all_points | INT | Puntos de prórroga. **Vacío en origen = 0** |
| prorroga | INT | 1 si ot_all_points > 0 |
| bench_points, biggest_lead, biggest_scoring_run, lead_changes, points_fast_break, points_from_turnovers, points_in_the_paint, points_second_chance, times_tied, timeouts_remaining, rebounds_team, turnovers_team | INT | Resto de estadísticas del partido, solo tipadas |
| season_wins, season_losses | INT | Récord de la fuente. **No usar como entrada** hasta confirmar si incluyen el propio partido (riesgo de fuga) |
| marca_fuera_rango | INT | Nº de valores imposibles detectados en la fila (pasan a NULL y se recalculan) |
| marca_score_recalculado | INT | 1 si team_score era imposible o ilegible y se ha recalculado |
| marca_score_incoherente | INT | 1 si team_score es válido pero no coincide con la suma de cuartos (se deja el dato) |
| n_recuperados_regla | INT | Nº de datos de la fila recuperados con reglas exactas (ver abajo) |
| n_huecos_antes_de_imputar | INT | Nº de datos que seguían vacíos y se han imputado con la media móvil |
| temporada | INT | Partición |

**Fuera de curated**: `coachId` (id de una persona), `gameLabel`, `gameSubLabel`, `seriesGameNumber`, `seed` y `gameDate`.

## `nba_curated.estadisticas_avanzadas`

Dos filas por partido, de `TeamStatisticsExtended.csv`. No se ensucia: se tipa, se quitan duplicados y se guardan solo las columnas que vamos a usar (ritmo, eficiencia y estilo). Tiene menos filas que `estadisticas_equipo` porque la fuente empieza más tarde.

| Columna | Tipo | Descripción |
|---|---|---|
| game_id, team_id, game_datetime, home | | Clave y contexto |
| offensive_rating, defensive_rating, net_rating | DOUBLE | Puntos por 100 posesiones a favor, en contra y diferencia |
| pace, possessions | DOUBLE | Ritmo (posesiones por 48 min) y posesiones. Clave para el objetivo 2 |
| effective_field_goal_percentage, true_shooting_percentage | DOUBLE | eFG% y TS% |
| assist_percentage, assist_to_turnover_ratio | DOUBLE | Juego de pase |
| offensive_rebound_percentage, defensive_rebound_percentage, rebound_percentage | DOUBLE | Rebote |
| team_turnover_percentage | DOUBLE | Pérdidas por posesión |
| free_throw_attempt_rate | DOUBLE | Tiros libres buscados |
| percent_field_goal_attempts_3point, percent_points_3point, percent_points_in_paint, percent_points_fast_break, percent_points_free_throw | DOUBLE | Estilo: de dónde salen los puntos |
| temporada | INT | Partición |

---

## `nba_curated.modelado_partidos`

Tabla **compacta para los modelos**: una fila por partido, **22 columnas** (la primera versión tenía 55). Se crea con `hive/04_modelado.hql` y se comprueba contra `scripts/construir_modelado.py`. Solo temporada regular desde 2000-01, partidos jugados y con 10 partidos previos de cada equipo (31.218 filas con los datos de Kaggle del 17/11).

**Regla anti-fuga:** cada entrada resume los 10 partidos **anteriores** de cada equipo (la ventana del HQL acaba en `1 PRECEDING`, el partido actual nunca entra). Por eso no hay ni marcador, ni estadísticas del propio partido, ni `season_wins/losses`, ni asistencia entre las entradas.

| Grupo | Columnas | Tipo | Qué es |
|---|---|---|---|
| Identificación (no son entradas) | game_id, game_date, home_team_id, away_team_id, temporada | | Para ordenar, cortar por fecha y volver al partido. Los ids **no** se dan al modelo como números |
| Forma (10 últimos) | dif_pts_favor_ult10, dif_pts_contra_ult10, dif_win_pct_ult10 | DOUBLE | Local − visitante de: puntos a favor, puntos en contra y % de victorias |
| Juego (10 últimos) | dif_tiro_campo_pct_ult10, dif_triples_pct_ult10, dif_libres_pct_ult10, dif_rebotes_ult10, dif_asistencias_ult10, dif_perdidas_ult10, dif_robos_ult10, dif_tapones_ult10 | DOUBLE | Local − visitante de cada estadística media |
| Calendario y temporada | dif_dias_descanso, dif_win_pct_temporada | DOUBLE | Descanso (tope 7 días) y % de victorias de la temporada antes del partido, local − visitante |
| Para el objetivo 2 | suma_pts_favor_ult10, suma_pts_contra_ult10 | DOUBLE | Local + visitante: indican si el partido será «de muchos puntos» |
| **Salida 1** | **home_win** | INT | 1 si gana el local |
| **Salida 2** | **puntos_totales** | INT | Puntos de los dos equipos (con prórrogas) |

### Por qué hay 15 entradas y no 48 (qué se quitó y por qué)

Medido con `scripts/comparar_columnas.py` (mismos modelos, prueba en 2023-24; resultados en `docs/seleccion_columnas_resultados.csv` y en MLflow, experimento `hoopmind_seleccion_columnas`).

| Qué se quitó | Columnas | Motivo |
|---|---|---|
| Ventana de 5 partidos de puntos a favor/en contra/margen/% victorias | 8 (`local_/visit_ …_ult5`) | Muy parecidas a la de 10 (correlación 0,92–0,93); quitarlas no empeora nada (AUC 0,711 → 0,712) |
| Margen (`margen_ult10`) | 2 | Es exactamente puntos a favor − puntos en contra: no añade información |
| `back_to_back` | 2 | Es `dias_descanso == 1`: ya está en `dif_dias_descanso` |
| `victorias_temporada` y `partidos_temporada` | 4 | `win_pct_temporada` ya es victorias ÷ partidos; `partidos_temporada` local y visitante se parecen un 99,7 % (los dos equipos van por el mismo nº de partido) |
| Pares `local_X` y `visit_X` | 28 → 13 `dif_X` | Al modelo le importa la **comparación** entre los dos equipos. Una diferencia sustituye a dos columnas |
| Se añaden | 2 sumas | Sin ellas, la regresión de puntos totales se queda en un error de ~26 puntos (casi como no usar entradas), porque las diferencias no dicen si el partido será de muchos o pocos puntos |

**Coste de reducir:** la regresión logística pasa de AUC 0,711 a 0,704 (−0,007) y el boosting de 0,706 a 0,700; el error de los puntos totales casi no cambia (14,78 → 14,83). Es la compensación elegida: 48 → 15 entradas con una pérdida muy pequeña, y un modelo más fácil de explicar. Las 48 siguen reproducibles con `comparar_columnas.py` por si el informe quiere defender la elección.

**Pendiente de confirmar:** `TeamStatisticsExtended` (ritmo, ratings) no se ha usado aún; si se añade, se mira de nuevo la selección.

---

## Reglas de limpieza (resumen)

| Problema (plan de ensuciado) | Regla en Hive |
|---|---|
| Duplicados | `ROW_NUMBER()` por `(game_id, team_id)` o `game_id`; se queda una fila |
| Fechas en varios formatos | Se reconoce el formato con `RLIKE` (`yyyy-MM-dd`, `dd/MM/yyyy`, `yyyy/MM/dd`, `MM-dd-yyyy`) y se parsea con su patrón. Lo que no se puede leer se cuenta y queda fuera |
| Tipos erróneos («112 pts», «0,451») | `regexp_replace`: coma → punto y fuera todo lo que no sea número |
| Valores imposibles | Rangos: puntos 0-200, % 0-1, conteos 0-200. Fuera de rango → NULL y se recupera con las reglas |
| Nulos | 1) Reglas exactas: anotados = % × intentados; intentados = anotados / %; % = anotados / intentados; rebotes totales = ofensivos + defensivos (y al revés); puntos = suma de cuartos o 2·TC + T3 + TL. 2) Si no se puede: media de los **10 partidos anteriores** del equipo y, si no, de todos sus partidos anteriores. Nunca la media de la temporada entera, porque incluiría partidos futuros |
| Nombres de equipo | Nombre más frecuente de cada `team_id` en cada temporada |
| Local y visitante al revés | Si Games dice local = X pero TeamStatistics dice que X jugó fuera, se intercambian ids, nombres y marcador |
| Marcador incoherente | Manda el marcador de TeamStatistics (ya validado con los cuartos) |
| Anomalías (idea A1) | Detector IQR de puntos totales por temporada en `03_comprobaciones.hql`. Se cuentan, no se borran |
