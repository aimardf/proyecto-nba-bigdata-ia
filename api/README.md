# API HoopMind (FastAPI)

Usa los dos modelos guardados por `scripts/modelos_base.py`: `models/modelo_ganador.joblib` y `models/modelo_puntos.joblib`.

## Probarla en local (desde la raíz del repo)
```bash
pip install -r api/requirements.txt
python scripts/modelos_base.py --tabla data/modelado_partidos.parquet    # solo si aún no existe models/
python scripts/estado_equipos.py --curated data/curated_export           # estado de cada equipo, para elegir dos equipos
uvicorn api.main:app --reload
```
- http://127.0.0.1:8000/ : página para elegir dos equipos (y un modo manual con las 15 entradas).
- http://127.0.0.1:8000/docs : pantalla de pruebas automática (cada endpoint tiene "Try it out").
- http://127.0.0.1:8000/health : comprueba que los modelos están cargados.

## Endpoints
| Método | Ruta | Qué hace |
|---|---|---|
| GET | `/health` | Estado de la API y de los dos modelos |
| GET | `/contrato` | Las 15 entradas, en orden |
| GET | `/equipos` | Equipos disponibles y fecha de los datos |
| POST | `/predict/partido` | Eliges local y visitante: calcula las 15 entradas y predice ganador y puntos |
| POST | `/predict/ganador` | Probabilidad de que gane el local |
| POST | `/predict/puntos` | Puntos totales (relativo a la media de liga; salida acotada a 127-319) |

Entradas (15 números, calculados antes del partido): `dif_pts_favor_ult10`, `dif_pts_contra_ult10`, `dif_win_pct_ult10`,
`dif_tiro_campo_pct_ult10`, `dif_triples_pct_ult10`, `dif_libres_pct_ult10`, `dif_rebotes_ult10`, `dif_asistencias_ult10`,
`dif_perdidas_ult10`, `dif_robos_ult10`, `dif_tapones_ult10`, `dif_dias_descanso`, `dif_win_pct_temporada`,
`suma_pts_favor_ult10`, `suma_pts_contra_ult10`. `/predict/puntos` admite además `media_liga_reciente` (por defecto 229,3).

Ejemplo:
```bash
curl -X POST http://127.0.0.1:8000/predict/ganador -H "Content-Type: application/json" \
  -d '{"dif_pts_favor_ult10":4,"dif_pts_contra_ult10":-3,"dif_win_pct_ult10":0.3,"dif_tiro_campo_pct_ult10":0.01,"dif_triples_pct_ult10":0.01,"dif_libres_pct_ult10":0,"dif_rebotes_ult10":2,"dif_asistencias_ult10":1,"dif_perdidas_ult10":-1,"dif_robos_ult10":0,"dif_tapones_ult10":0,"dif_dias_descanso":1,"dif_win_pct_temporada":0.2,"suma_pts_favor_ult10":228,"suma_pts_contra_ult10":226}'
```

## Pruebas
```bash
python -m pytest api/test_api.py -q      # usan modelos de juguete, no hacen falta datos reales
```
Si faltan los modelos, los endpoints de predicción responden 503 con el comando para generarlos.

## Elegir dos equipos
`scripts/estado_equipos.py` guarda en `models/estado_equipos.json` la media de los últimos 10 partidos de cada equipo y su % de victorias de la temporada, con las mismas definiciones que `construir_modelado.py` (hay una prueba que lo comprueba). Los datos llegan hasta la fecha del último partido; para actualizarlos hay que volver a ejecutar el script con datos nuevos. Si en tu PC no existe `data/curated_export`, usa `--raw C:/Users/aimar/Desktop/nba-datos`.
