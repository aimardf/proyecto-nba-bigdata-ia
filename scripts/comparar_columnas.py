#!/usr/bin/env python3
"""comparar_columnas.py · HoopMind

¿Sobran columnas en la tabla de modelado? Este script lo comprueba con datos, no a ojo:
construye la tabla ANCHA de la primera versión (48 entradas) y la compara con versiones
más compactas, entrenando los mismos modelos y midiendo en el mismo trozo de prueba (temporada 2023-24).

Cada variante se guarda como un "run" de MLflow (experimento hoopmind_seleccion_columnas) y todo queda en
docs/seleccion_columnas_resultados.csv, junto con las columnas casi duplicadas (correlación > 0,9).

Uso:
  python3 scripts/comparar_columnas.py --datos ~/Desktop/nba-datos   # Games.csv y TeamStatistics.csv ORIGINALES
"""
import argparse
import pathlib
import re

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as HGBC, HistGradientBoostingRegressor as HGBR
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

STATS = ["fieldGoalsPercentage", "threePointersPercentage", "freeThrowsPercentage",
         "reboundsTotal", "assists", "turnovers", "steals", "blocks"]


def tabla_ancha(datos):
    """Tabla de la primera versión: ventanas de 5 y 10, local_/visit_/dif_ (48 entradas)."""
    g = pd.read_csv(datos / "Games.csv", low_memory=False)
    ts = pd.read_csv(datos / "TeamStatistics.csv", low_memory=False, usecols=["gameId", "teamId"] + STATS)
    g["fecha"] = pd.to_datetime(g.gameDateTimeEst).dt.normalize()
    g["temporada"] = np.where(g.fecha.dt.month >= 8, g.fecha.dt.year, g.fecha.dt.year - 1)
    g = g[(g.gameType == "Regular Season") & (g.temporada >= 2000) & (g.homeScore > 0) & (g.awayScore > 0)]
    g = g.sort_values(["fecha", "gameId"]).reset_index(drop=True)

    def lado(a, b, es):
        return pd.DataFrame({"gameId": g.gameId, "fecha": g.fecha, "temporada": g.temporada, "equipo": g[a + "teamId"],
                             "es_local": es, "pts_favor": g[a + "Score"], "pts_contra": g[b + "Score"]})
    L = pd.concat([lado("home", "away", 1), lado("away", "home", 0)]).sort_values(["equipo", "fecha", "gameId"])
    L = L.reset_index(drop=True)
    L["gana"] = (L.pts_favor > L.pts_contra).astype(int)
    L["margen"] = L.pts_favor - L.pts_contra
    L = L.merge(ts.rename(columns={"teamId": "equipo"}), on=["gameId", "equipo"], how="left")
    L = L.sort_values(["equipo", "fecha", "gameId"]).reset_index(drop=True)
    grp = L.groupby("equipo", group_keys=False)
    roll = lambda c, n: grp[c].transform(lambda s: s.shift(1).rolling(n, min_periods=n).mean())
    for n in (5, 10):
        for c, nom in [("pts_favor", "pts_favor"), ("pts_contra", "pts_contra"), ("margen", "margen"), ("gana", "win_pct")]:
            L[f"{nom}_ult{n}"] = roll(c, n)
    for c in STATS:
        L[f"{c}_ult10"] = roll(c, 10)
    L["dias_descanso"] = grp["fecha"].transform(lambda s: s.diff().dt.days).clip(upper=7)
    L["back_to_back"] = (L.dias_descanso == 1).astype(int)
    gt = L.groupby(["equipo", "temporada"], group_keys=False)
    L["partidos_temporada"] = gt.cumcount()
    L["victorias_temporada"] = gt["gana"].transform(lambda s: s.shift(1).cumsum()).fillna(0)
    L["win_pct_temporada"] = (L.victorias_temporada / L.partidos_temporada).where(L.partidos_temporada > 0)
    met = [c for c in L.columns if c.endswith(("_ult5", "_ult10"))] + \
          ["dias_descanso", "back_to_back", "partidos_temporada", "victorias_temporada", "win_pct_temporada"]
    met = [c for c in met if c not in STATS]
    lo = L[L.es_local == 1].set_index("gameId")[met].add_prefix("local_")
    vi = L[L.es_local == 0].set_index("gameId")[met].add_prefix("visit_")
    ds = g.set_index("gameId")[["fecha", "temporada", "homeScore", "awayScore", "winner", "hometeamId"]].join(lo).join(vi)
    for m in met:
        ds[f"dif_{m}"] = ds[f"local_{m}"] - ds[f"visit_{m}"]
    ds["suma_pts_favor_ult10"] = ds.local_pts_favor_ult10 + ds.visit_pts_favor_ult10
    ds["suma_pts_contra_ult10"] = ds.local_pts_contra_ult10 + ds.visit_pts_contra_ult10
    ds["home_win"] = (ds.winner == ds.hometeamId).astype(int)
    ds["puntos_totales"] = ds.homeScore + ds.awayScore
    return ds.dropna(subset=["local_pts_favor_ult10", "visit_pts_favor_ult10"]).reset_index()


def evaluar(ds, cols):
    tr, te = ds.temporada <= 2022, ds.temporada == 2023
    pre = lambda m: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), m)
    r = {}
    y = ds.home_win
    r["auc_logistica"] = roc_auc_score(y[te], pre(LogisticRegression(max_iter=2000)).fit(ds.loc[tr, cols], y[tr]).predict_proba(ds.loc[te, cols])[:, 1])
    r["auc_boosting"] = roc_auc_score(y[te], HGBC(max_iter=200, learning_rate=.05, random_state=0).fit(ds.loc[tr, cols], y[tr]).predict_proba(ds.loc[te, cols])[:, 1])
    y = ds.puntos_totales
    r["mae_ridge"] = mean_absolute_error(y[te], pre(Ridge(10)).fit(ds.loc[tr, cols], y[tr]).predict(ds.loc[te, cols]))
    r["mae_boosting"] = mean_absolute_error(y[te], HGBR(max_iter=200, learning_rate=.05, random_state=0).fit(ds.loc[tr, cols], y[tr]).predict(ds.loc[te, cols]))
    return {k: round(float(v), 4) for k, v in r.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datos", required=True)
    ap.add_argument("--csv", default="docs/seleccion_columnas_resultados.csv")
    ap.add_argument("--mlflow", default="sqlite:///mlflow.db")
    a = ap.parse_args()
    ds = tabla_ancha(pathlib.Path(a.datos))
    # la versión ancha de la primera entrega: local_ y visit_ de todo + 6 diferencias (48 entradas)
    dif6 = ["dif_" + m for m in ("pts_favor_ult10", "pts_contra_ult10", "margen_ult10", "win_pct_ult10",
                                 "dias_descanso", "win_pct_temporada")]
    X48 = [c for c in ds.columns if c.startswith(("local_", "visit_"))] + dif6
    print("Partidos:", len(ds), "| entradas de la versión ancha:", len(X48))

    # columnas casi duplicadas
    C = ds[X48].corr().abs()
    pares = sorted([(x, y, C.loc[x, y]) for i, x in enumerate(X48) for y in X48[i + 1:] if C.loc[x, y] > 0.9], key=lambda p: -p[2])
    print("Pares con |r| > 0,9:")
    for x, y, r in pares:
        print(f"  {x} ~ {y}: {r:.3f}")

    base = [c for c in ds.columns if c.startswith("dif_") and "margen" not in c and "_ult5" not in c
            and not c.endswith(("back_to_back", "partidos_temporada", "victorias_temporada"))]
    variantes = {
        "A_todas_48": X48,
        "B_sin_ventana5": [c for c in X48 if "_ult5" not in c],
        "C_B_sin_back2back_ni_victorias": [c for c in X48 if "_ult5" not in c and not c.endswith(("back_to_back", "victorias_temporada"))],
        "D_solo_diferencias_sin_sumas": base,
        "E_FINAL_diferencias_y_sumas": base + ["suma_pts_favor_ult10", "suma_pts_contra_ult10"],
    }
    mlflow.set_tracking_uri(a.mlflow)
    mlflow.set_experiment("hoopmind_seleccion_columnas")
    filas = []
    for nombre, cols in variantes.items():
        m = evaluar(ds, cols)
        with mlflow.start_run(run_name=nombre):
            mlflow.log_param("n_entradas", len(cols))
            mlflow.log_param("columnas", ",".join(cols))
            mlflow.log_metrics(m)
        filas.append({"variante": nombre, "n_entradas": len(cols), **m})
        print(nombre, len(cols), m)
    out = pathlib.Path(a.csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(filas).to_csv(out, index=False)
    print("Guardado:", out)


if __name__ == "__main__":
    main()
