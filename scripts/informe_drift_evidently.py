#!/usr/bin/env python3
"""informe_drift_evidently.py · HoopMind

Informe de DRIFT con Evidently (lo pide el enunciado). Compara dos grupos de partidos:
  referencia  = lo que vio el modelo al entrenar: temporadas <= 2022 (2000-01 a 2022-23)
  actual      = lote 2: temporadas >= 2024 (2024-25 en adelante)
(la temporada 2023 es la de prueba y se deja fuera de los dos grupos, igual que en modelos_base.py)

Genera, en docs/drift/:
  informe_drift_entradas_y_objetivo.html   las 15 entradas + puntos_totales (objetivo 2, absoluto)
  informe_drift_absoluto_vs_relativo.html  solo el objetivo: puntos ABSOLUTOS frente a puntos RELATIVOS a la liga
  resumen_drift_evidently.csv              tabla propia con PSI y KS de cada columna (comprobación independiente)

Método de drift: PSI (el que pide el enunciado). Regla habitual: PSI < 0,1 estable, 0,1-0,25 cambio moderado, > 0,25 drift.
El objetivo RELATIVO es el del PR 11: puntos_totales - media de liga de los 600 partidos anteriores (sin año).

Funciona con Evidently nuevo (0.7.x: evidently.Report) y con el antiguo (0.4-0.6: evidently.report.Report).

Uso:
  python scripts/informe_drift_evidently.py --tabla data/modelado_partidos.parquet
  python scripts/informe_drift_evidently.py --sintetico        # prueba rápida con datos inventados
"""
import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from modelo_puntos import media_liga_reciente  # noqa: E402

IDENT = ["game_id", "game_date", "temporada", "home_team_id", "away_team_id"]
SALIDAS = ["home_win", "puntos_totales"]
OBJ_ABS, OBJ_REL = "puntos_totales", "puntos_relativos_liga"


def psi(ref, cur, bins=10):
    """PSI: compara cómo se reparten los valores en 'bins' tramos (los tramos salen de la referencia).
    Valores fuera del rango de referencia caen en los tramos de los extremos (por eso crece mucho si se desplaza todo)."""
    ref, cur = pd.Series(ref).dropna().values, pd.Series(cur).dropna().values
    cortes = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    cortes[0], cortes[-1] = -np.inf, np.inf
    p = np.histogram(ref, cortes)[0] / len(ref)
    q = np.histogram(cur, cortes)[0] / len(cur)
    p, q = np.clip(p, 1e-4, None), np.clip(q, 1e-4, None)
    return float(np.sum((q - p) * np.log(q / p)))


def tabla_resumen(ref, cur, columnas):
    from scipy.stats import ks_2samp
    filas = []
    for c in columnas:
        v = psi(ref[c], cur[c])
        ks = ks_2samp(ref[c].dropna(), cur[c].dropna())
        filas.append({"columna": c, "media_referencia": ref[c].mean(), "media_actual": cur[c].mean(),
                      "psi": v, "ks": ks.statistic, "p_ks": ks.pvalue,
                      "veredicto": "DRIFT" if v > .25 else ("moderado" if v > .1 else "estable")})
    return pd.DataFrame(filas).sort_values("psi", ascending=False).round(4)


def informe(ref, cur, columnas, ruta):
    """Crea y guarda un informe de Evidently (PSI) con las columnas dadas. Sirve para las dos versiones de Evidently."""
    try:                                                # Evidently nuevo (0.7.x)
        from evidently import DataDefinition, Dataset, Report
        from evidently.presets import DataDriftPreset
        dd = DataDefinition(numerical_columns=columnas)
        ds_ref = Dataset.from_pandas(ref[columnas], data_definition=dd)
        ds_cur = Dataset.from_pandas(cur[columnas], data_definition=dd)
        snap = Report([DataDriftPreset(method="psi")]).run(ds_cur, ds_ref)
        snap.save_html(str(ruta))
    except ImportError:                                 # Evidently antiguo (0.4 - 0.6)
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report
        rep = Report(metrics=[DataDriftPreset(stattest="psi")])
        rep.run(reference_data=ref[columnas], current_data=cur[columnas])
        rep.save_html(str(ruta))
    print(f"  guardado {ruta}")


def datos_sinteticos(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    temp = rng.integers(2000, 2026, n)
    temp[temp == 2023] = 2022
    ds = pd.DataFrame({"game_id": np.arange(n), "temporada": temp,
                       "game_date": pd.to_datetime("2000-10-01") + pd.to_timedelta((temp - 2000) * 365 + rng.integers(0, 200, n), "D"),
                       "home_team_id": 1, "away_team_id": 2, "home_win": rng.integers(0, 2, n)})
    nivel = 200 + (temp - 2000) * 1.1
    ds["puntos_totales"] = rng.normal(nivel, 18)
    for c in ["dif_pts_favor_ult10", "dif_win_pct_ult10", "dif_dias_descanso"]:
        ds[c] = rng.normal(0, 1, n)
    ds["suma_pts_favor_ult10"] = rng.normal(nivel, 6)
    ds["suma_pts_contra_ult10"] = rng.normal(nivel, 6)
    return ds.sort_values("game_date").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tabla", default="data/modelado_partidos.parquet")
    ap.add_argument("--sintetico", action="store_true", help="usa datos inventados (solo para probar el script)")
    ap.add_argument("--fin-train", type=int, default=2022)
    ap.add_argument("--prueba", type=int, default=2023)
    ap.add_argument("--salida", default="docs/drift")
    a = ap.parse_args()

    ds = datos_sinteticos() if a.sintetico else pd.read_parquet(a.tabla)
    ds = ds.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    entradas = [c for c in ds.columns if c not in IDENT + SALIDAS]
    ds[OBJ_REL] = ds[OBJ_ABS] - media_liga_reciente(ds)        # objetivo relativo (PR 11)
    ref = ds[ds.temporada <= a.fin_train]
    cur = ds[ds.temporada > a.prueba]
    print(f"{len(entradas)} entradas | referencia {len(ref):,} partidos (<= {a.fin_train}) | actual {len(cur):,} (> {a.prueba})")
    if len(cur) == 0:
        sys.exit("No hay partidos del lote 2: revisa --prueba o la tabla.")

    out = pathlib.Path(a.salida)
    out.mkdir(parents=True, exist_ok=True)
    print("Informes de Evidently:")
    informe(ref, cur, entradas + [OBJ_ABS], out / "informe_drift_entradas_y_objetivo.html")
    informe(ref, cur, [OBJ_ABS, OBJ_REL], out / "informe_drift_absoluto_vs_relativo.html")

    res = tabla_resumen(ref, cur, entradas + [OBJ_ABS, OBJ_REL])
    res.to_csv(out / "resumen_drift_evidently.csv", index=False)
    pd.set_option("display.width", 200)
    print("\nPSI y KS por columna (de mayor a menor drift):")
    print(res[["columna", "media_referencia", "media_actual", "psi", "veredicto"]].to_string(index=False))
    r = res.set_index("columna").psi
    print(f"\nObjetivo ABSOLUTO: PSI {r[OBJ_ABS]:.2f}  |  objetivo RELATIVO a la liga: PSI {r[OBJ_REL]:.2f}")
    print(f"Columnas con drift (PSI > 0,25): {int((res.psi > .25).sum())} de {len(res)}")
    print(f"\nAbre los .html de {out}/ con doble clic.")


if __name__ == "__main__":
    main()
