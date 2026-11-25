#!/usr/bin/env python3
"""estado_equipos.py · HoopMind

Guarda el ESTADO ACTUAL de cada equipo para poder elegir "Equipo A contra Equipo B" en la API:
la media de sus últimos 10 partidos (puntos a favor y en contra, % de victorias y las 8 estadísticas de juego)
y su % de victorias de la temporada. Usa exactamente las mismas definiciones que construir_modelado.py,
solo que sin dejar fuera el último partido (ahora no hay partido que predecir "dentro" de los datos).

También guarda la media de liga reciente (media de puntos totales de los últimos 600 partidos jugados),
que necesita el modelo de puntos.

Uso (desde la raíz del repo; las mismas carpetas que construir_modelado.py):
  python3 scripts/estado_equipos.py --curated data/curated_export
  python3 scripts/estado_equipos.py --raw C:/Users/aimar/Desktop/nba-datos       # alternativa con los CSV originales
Salida por defecto: models/estado_equipos.json

Además guarda models/historial_equipos.csv.gz: una fila por equipo y partido (temporada regular desde 2000) con
sus puntos y estadísticas. La API lo usa para "viajar en el tiempo": a una fecha pasada le calcula la forma de cada
equipo con sus 10 partidos anteriores a esa fecha (misma definición que construir_modelado.py) y compara con el
resultado real. Pesa ~3 MB y es el único histórico que hace falta guardar.
"""
import argparse
import json
import pathlib

import numpy as np
import pandas as pd
from construir_modelado import STATS, VENTANA, cargar_curated, cargar_raw, largo_equipos
from modelo_puntos import VENT_LIGA, media_liga_actual



def _num(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)


def _nombres(g):
    """team_id -> 'Ciudad Nombre' con el último partido en que aparece (curated: home_team_city; raw: hometeam_city)."""
    pares = []
    for lado, ident in (("home", "home_team_id"), ("away", "away_team_id")):
        for pref in (f"{lado}_team_", f"{lado}team_"):
            if f"{pref}name" in g.columns:
                ciudad = g[f"{pref}city"] if f"{pref}city" in g.columns else ""
                pares.append(pd.DataFrame({"equipo": g[ident], "game_date": g.game_date,
                                           "nombre": (ciudad.astype(str) + " " + g[f"{pref}name"].astype(str)).str.strip()}))
                break
    if not pares:
        return {}
    n = pd.concat(pares).sort_values("game_date").drop_duplicates("equipo", keep="last")
    return dict(zip(n.equipo, n.nombre))


COLS_HISTORIAL = ["game_id", "game_date", "temporada", "equipo", "rival", "es_local", "pts_favor", "pts_contra", "gana"] + list(STATS.values())
FUENTES = {"pts_favor": "pts_favor", "pts_contra": "pts_contra", "win_pct": "gana", **{v: v for v in STATS.values()}}


def temporada_de(fecha):
    """Temporada NBA a la que pertenece una fecha (la temporada 2024 empieza en octubre de 2024)."""
    fecha = pd.Timestamp(fecha)
    return fecha.year if fecha.month >= 8 else fecha.year - 1


def historial(largo):
    """Tabla larga + columna 'rival' (el otro equipo del mismo partido), lista para guardar."""
    par = largo[["game_id", "equipo"]].rename(columns={"equipo": "rival"})
    h = largo.merge(par, on="game_id")
    h = h[h.equipo != h.rival].drop_duplicates(["game_id", "equipo"])
    h = h.sort_values(["game_date", "game_id", "equipo"])[COLS_HISTORIAL].reset_index(drop=True)
    cols = list(STATS.values())
    h[cols] = h[cols].round(4)
    return h


def forma_hasta(h, equipo, fecha):
    """Forma de un equipo ANTES de 'fecha' (no cuenta ese día): media de sus últimos 10 partidos, % de victorias de
    esa temporada y días de descanso. Devuelve el mismo formato que cada equipo de estado_equipos.json, o None si
    el equipo no tiene 10 partidos previos. 'h' es la tabla de historial()."""
    fecha = pd.Timestamp(fecha)
    previos = h[(h.equipo == equipo) & (h.game_date < fecha)].sort_values(["game_date", "game_id"])
    if len(previos) < VENTANA:
        return None
    ult = previos.tail(VENTANA)
    temp = previos[previos.temporada == temporada_de(fecha)]
    fila = {"id": int(equipo), "ultimo_partido": str(previos.game_date.max().date()), "partidos_temporada": int(len(temp)),
            "win_pct_temporada": _num(temp.gana.mean()) if len(temp) else None,
            "dias_descanso": int(min(7, (fecha - previos.game_date.max()).days))}
    for nombre, col in FUENTES.items():
        fila[f"{nombre}_ult{VENTANA}"] = _num(ult[col].mean())
    return fila


def calcular(g, e):
    gf, largo = largo_equipos(g, e)
    ultima = int(largo.temporada.max())
    fuentes = FUENTES
    nombres = _nombres(g)
    equipos = []
    for eq, d in largo[largo.temporada == ultima].groupby("equipo"):
        hist = largo[largo.equipo == eq].tail(VENTANA)          # últimos 10 partidos del equipo (cualquier temporada)
        if len(hist) < VENTANA:
            continue
        temp = d  # partidos de la última temporada
        fila = {"id": int(eq), "nombre": nombres.get(eq, f"Equipo {int(eq)}"),
                "ultimo_partido": str(d.game_date.max().date()), "partidos_temporada": int(len(temp)),
                "win_pct_temporada": _num(temp.gana.mean())}
        for nombre, col in fuentes.items():
            fila[f"{nombre}_ult{VENTANA}"] = _num(hist[col].mean())
        equipos.append(fila)
    equipos.sort(key=lambda x: x["nombre"])
    totales = (gf.home_score + gf.away_score).values[np.argsort(gf.game_date.values, kind="stable")]
    return {"fecha_corte": str(gf.game_date.max().date()), "temporada": ultima, "ventana": VENTANA,
            "media_liga_reciente": round(media_liga_actual(totales), 2), "ventana_liga": VENT_LIGA, "equipos": equipos}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--curated", help="carpeta con partidos.csv y estadisticas_equipo.csv")
    ap.add_argument("--raw", help="carpeta con Games.csv y TeamStatistics.csv originales")
    ap.add_argument("--salida", default="models/estado_equipos.json")
    ap.add_argument("--historial", default="models/historial_equipos.csv.gz", help="histórico por equipo para predecir fechas pasadas")
    a = ap.parse_args()
    if bool(a.curated) == bool(a.raw):
        ap.error("indica --curated o --raw (uno solo)")
    g, e = cargar_curated(pathlib.Path(a.curated)) if a.curated else cargar_raw(pathlib.Path(a.raw))
    est = calcular(g, e)
    hist = historial(largo_equipos(g, e)[1])
    pathlib.Path(a.historial).parent.mkdir(parents=True, exist_ok=True)
    hist.to_csv(a.historial, index=False, compression="gzip")
    print(f"Historial: {len(hist):,} filas equipo-partido -> {a.historial}")
    out = pathlib.Path(a.salida)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(est, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(est['equipos'])} equipos de la temporada {est['temporada']} | datos hasta {est['fecha_corte']} "
          f"| media de liga reciente {est['media_liga_reciente']} -> {out}")


if __name__ == "__main__":
    main()
