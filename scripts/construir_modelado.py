#!/usr/bin/env python3
"""construir_modelado.py · HoopMind

Construye la tabla de MODELADO (una fila por partido, compacta) a partir de la zona curated.
Hace lo mismo que hive/04_modelado.hql; sirve para (a) trabajar sin Hive y (b) COMPROBAR el
resultado de Hive (opción --hive).

Regla anti-fuga: todas las entradas son resúmenes de partidos ANTERIORES (shift(1) y después media
de los últimos 10). El partido que se predice nunca entra en su propio resumen.

Uso:
  # desde la exportación de curated que deja pipeline_limpieza.sh (partidos.csv y estadisticas_equipo.csv)
  python3 scripts/construir_modelado.py --curated data/curated_export --salida data/modelado_partidos.parquet
  # comprobar que Hive y Python dan lo mismo
  python3 scripts/construir_modelado.py --curated data/curated_export --hive data/curated_export/modelado_partidos.csv
  # demostración con los CSV originales de Kaggle (sin ensuciar), si aún no hay curated
  python3 scripts/construir_modelado.py --raw ~/Desktop/nba-datos --salida data/modelado_raw.parquet
"""
import argparse
import pathlib
import re

import numpy as np
import pandas as pd

VENTANA = 10
# estadística de curated.estadisticas_equipo -> nombre corto en español (para la tabla de modelado)
STATS = {
    "field_goals_percentage": "tiro_campo_pct",
    "three_pointers_percentage": "triples_pct",
    "free_throws_percentage": "libres_pct",
    "rebounds_total": "rebotes",
    "assists": "asistencias",
    "turnovers": "perdidas",
    "steals": "robos",
    "blocks": "tapones",
}
# Entradas finales: diferencia local - visitante de cada resumen, y la suma de puntos a favor/en contra.
BASE_DIF = ["pts_favor", "pts_contra", "win_pct"] + list(STATS.values())
ENTRADAS = [f"dif_{m}_ult{VENTANA}" for m in BASE_DIF] + ["dif_dias_descanso", "dif_win_pct_temporada",
                                                          f"suma_pts_favor_ult{VENTANA}", f"suma_pts_contra_ult{VENTANA}"]
IDENT = ["game_id", "game_date", "temporada", "home_team_id", "away_team_id"]
SALIDAS = ["home_win", "puntos_totales"]
COLUMNAS = IDENT + ENTRADAS + SALIDAS


def snake(c):
    return re.sub(r"(?<!^)(?=[A-Z])", "_", c).lower()


def cargar_raw(carpeta):
    """CSV originales de Kaggle -> mismos nombres que curated (solo para demostrar)."""
    g = pd.read_csv(carpeta / "Games.csv", low_memory=False).rename(columns=snake)
    g = g.rename(columns={"hometeam_id": "home_team_id", "awayteam_id": "away_team_id",
                          "game_date_time_est": "game_datetime"})
    g["game_date"] = pd.to_datetime(g["game_datetime"]).dt.normalize()
    g["temporada"] = np.where(g.game_date.dt.month >= 8, g.game_date.dt.year, g.game_date.dt.year - 1)
    g["jugado"] = ((g.home_score > 0) & (g.away_score > 0)).astype(int)
    cols = ["game_id", "team_id"] + list(STATS)
    e = pd.read_csv(carpeta / "TeamStatistics.csv", low_memory=False).rename(columns=snake)[cols]
    return g, e


def cargar_curated(carpeta):
    g = pd.read_csv(carpeta / "partidos.csv", low_memory=False)
    g["game_date"] = pd.to_datetime(g["game_date"])
    e = pd.read_csv(carpeta / "estadisticas_equipo.csv", low_memory=False)[["game_id", "team_id"] + list(STATS)]
    return g, e


def construir(g, e):
    g = g[(g.game_type == "Regular Season") & (g.temporada >= 2000) & (g.jugado == 1)]
    g = g.sort_values(["game_date", "game_id"]).reset_index(drop=True)

    def lado(eq, rival, es_local, a_favor, en_contra):
        return pd.DataFrame({"game_id": g.game_id, "game_date": g.game_date, "temporada": g.temporada,
                             "equipo": g[eq], "es_local": es_local, "pts_favor": g[a_favor], "pts_contra": g[en_contra]})

    largo = pd.concat([lado("home_team_id", "away_team_id", 1, "home_score", "away_score"),
                       lado("away_team_id", "home_team_id", 0, "away_score", "home_score")])
    largo = largo.sort_values(["equipo", "game_date", "game_id"]).reset_index(drop=True)
    largo["gana"] = (largo.pts_favor > largo.pts_contra).astype(int)
    largo = largo.merge(e.rename(columns={"team_id": "equipo", **STATS}), on=["game_id", "equipo"], how="left") \
                 .sort_values(["equipo", "game_date", "game_id"]).reset_index(drop=True)

    grp = largo.groupby("equipo", group_keys=False)
    fuentes = {"pts_favor": "pts_favor", "pts_contra": "pts_contra", "win_pct": "gana",
               **{v: v for v in STATS.values()}}
    for nombre, col in fuentes.items():
        largo[f"{nombre}_ult{VENTANA}"] = grp[col].transform(
            lambda s: s.shift(1).rolling(VENTANA, min_periods=VENTANA).mean())
    largo["dias_descanso"] = grp["game_date"].transform(lambda s: s.diff().dt.days).clip(upper=7)
    gt = largo.groupby(["equipo", "temporada"], group_keys=False)
    partidos_prev = gt.cumcount()
    victorias_prev = gt["gana"].transform(lambda s: s.shift(1).cumsum()).fillna(0)
    largo["win_pct_temporada"] = (victorias_prev / partidos_prev).where(partidos_prev > 0)

    metricas = [f"{m}_ult{VENTANA}" for m in fuentes] + ["dias_descanso", "win_pct_temporada"]
    lo = largo[largo.es_local == 1].set_index("game_id")[metricas].add_prefix("local_")
    vi = largo[largo.es_local == 0].set_index("game_id")[metricas].add_prefix("visit_")
    ds = g.set_index("game_id")[["game_date", "temporada", "home_team_id", "away_team_id", "home_score", "away_score"]] \
          .join(lo).join(vi)
    for m in metricas:
        ds[f"dif_{m}"] = ds[f"local_{m}"] - ds[f"visit_{m}"]
    for m in ("pts_favor", "pts_contra"):
        ds[f"suma_{m}_ult{VENTANA}"] = ds[f"local_{m}_ult{VENTANA}"] + ds[f"visit_{m}_ult{VENTANA}"]
    ds["home_win"] = (ds.home_score > ds.away_score).astype(int)
    ds["puntos_totales"] = ds.home_score + ds.away_score
    ds = ds.dropna(subset=[f"local_pts_favor_ult{VENTANA}", f"visit_pts_favor_ult{VENTANA}"]).reset_index()
    return ds[COLUMNAS]


def comparar_con_hive(py, ruta):
    h = pd.read_csv(ruta, low_memory=False)
    h["game_date"] = pd.to_datetime(h["game_date"])
    m = py.merge(h, on="game_id", suffixes=("_py", "_hv"), how="outer", indicator=True)
    print("Filas Python:", len(py), "| Hive:", len(h), "| solo en uno de los dos:", int((m._merge != "both").sum()))
    peor = 0.0
    for c in ENTRADAS + SALIDAS:
        a, b = m[c + "_py"], m[c + "_hv"]
        d = np.nanmax(np.abs(a - b)) if len(m) else 0
        peor = max(peor, 0 if np.isnan(d) else d)
        if d > 1e-6:
            print(f"  DIFERENCIA en {c}: {d:.6f}")
    print("Máxima diferencia Hive vs Python:", f"{peor:.2e}", "-> OK" if peor < 1e-6 else "-> REVISAR")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--curated", help="carpeta con partidos.csv y estadisticas_equipo.csv")
    ap.add_argument("--raw", help="carpeta con Games.csv y TeamStatistics.csv originales (demostración)")
    ap.add_argument("--salida", default="data/modelado_partidos.parquet")
    ap.add_argument("--hive", help="CSV exportado de nba_curated.modelado_partidos para comparar")
    a = ap.parse_args()
    if bool(a.curated) == bool(a.raw):
        ap.error("indica --curated o --raw (uno solo)")
    g, e = cargar_curated(pathlib.Path(a.curated)) if a.curated else cargar_raw(pathlib.Path(a.raw))
    ds = construir(g, e)
    print(f"Tabla de modelado: {ds.shape[0]:,} partidos x {ds.shape[1]} columnas "
          f"({len(IDENT)} identificación + {len(ENTRADAS)} entradas + {len(SALIDAS)} salidas)")
    if a.hive:
        comparar_con_hive(ds, a.hive)
    else:
        out = pathlib.Path(a.salida)
        out.parent.mkdir(parents=True, exist_ok=True)
        ds.to_parquet(out, index=False, compression="snappy")
        print("Guardado en", out)


if __name__ == "__main__":
    main()
