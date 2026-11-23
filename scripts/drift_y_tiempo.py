#!/usr/bin/env python3
"""drift_y_tiempo.py · HoopMind

Drift del objetivo 2 (puntos_totales) y qué hacer con la variable tiempo.

  1. MIDE el drift: media de puntos por temporada, PSI y KS de cada entrada y del objetivo
     (entrenamiento antiguo vs temporadas recientes) y cuánto sube el error.
  2. COMPARA estrategias con validación temporal (entrena antiguo, prueba reciente) y copia de cada una a MLflow:
        A  actual                 15 entradas absolutas, objetivo absoluto (lo que ya teníamos)
        B  + año como número      el error que queremos evitar (B2: sin resúmenes de puntos, el año hace de tendencia)
        C  entradas relativas     los resúmenes de puntos se restan de la media de liga reciente
        D  C + objetivo relativo  se predice (puntos - media de liga reciente) y se vuelve a sumar
        E  D + ventana reciente   entrenar solo con las últimas N temporadas
        F  D + pesos por recencia partidos recientes pesan más (vida media de H temporadas)
        G  mejor + clip           acotar la predicción a un rango realista
  3. PRUEBA DE ESTRÉS "2050": ¿qué predice cada modelo si le pedimos el año 2050 o si la liga anotara mucho más?
  4. REENTRENO: modelo fijo (entrenado una vez) frente a reentrenado cada temporada.

Salidas: docs/drift/*.csv, docs/drift/*.png y runs en MLflow (experimento hoopmind_obj2_drift).

Uso:
  python3 scripts/drift_y_tiempo.py --tabla data/modelado_partidos.parquet
"""
import argparse
import pathlib
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import HistGradientBoostingRegressor as HGBR
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
IDENT = ["game_id", "game_date", "temporada", "home_team_id", "away_team_id"]
SALIDAS = ["home_win", "puntos_totales"]
SUMAS = ["suma_pts_favor_ult10", "suma_pts_contra_ult10"]   # únicas entradas con nivel absoluto de puntos
AZUL, GRANATE, CREMA, NARANJA = "#174D75", "#21060A", "#F8F5E4", "#D9822B"
VENT_LIGA = 600          # partidos previos que forman la "media de liga reciente" (~ un mes y medio de temporada)
FIN_TRAIN, PRUEBA = 2022, 2023   # mismo corte que modelos_base.py; lote 2 = temporadas >= 2024


# ---------------------------------------------------------------- utilidades
def media_liga_reciente(ds, n=VENT_LIGA):
    """Media de puntos totales de los n partidos ANTERIORES (por fecha; sin fuga: el propio día no cuenta)."""
    d = ds.groupby("game_date").puntos_totales.agg(["size", "sum"]).sort_index()
    cn, cs = d["size"].cumsum().values, d["sum"].cumsum().values
    prev_n = np.r_[0, cn[:-1]]; prev_s = np.r_[0, cs[:-1]]
    ref = np.empty(len(d))
    for i in range(len(d)):
        if prev_n[i] == 0:
            ref[i] = np.nan; continue
        ini = np.searchsorted(prev_n, max(prev_n[i] - n, 0), side="left")
        ref[i] = (prev_s[i] - prev_s[ini]) / max(prev_n[i] - prev_n[ini], 1)
    r = pd.Series(ref, index=d.index).bfill()        # primer día: usa el siguiente valor disponible
    return ds.game_date.map(r).values


def psi(ref, act, bins=10):
    """Population Stability Index con deciles del periodo de referencia. <0.1 estable, 0.1-0.25 vigilar, >0.25 drift."""
    ref, act = pd.Series(ref).dropna(), pd.Series(act).dropna()
    cortes = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    cortes[0], cortes[-1] = -np.inf, np.inf
    p = np.histogram(ref, cortes)[0] / len(ref); q = np.histogram(act, cortes)[0] / len(act)
    p, q = np.clip(p, 1e-4, None), np.clip(q, 1e-4, None)
    return float(np.sum((q - p) * np.log(q / p)))


def metr(y, p):
    return {"mae": mean_absolute_error(y, p), "rmse": float(np.sqrt(mean_squared_error(y, p))),
            "r2": r2_score(y, p), "sesgo": float(np.mean(p - y))}


def crear(tipo):
    if tipo == "ridge":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10))
    return HGBR(max_iter=200, learning_rate=.05, random_state=0)


def ajustar(m, X, y, w=None):
    if w is None:
        m.fit(X, y)
    elif hasattr(m, "steps"):
        m.fit(X, y, ridge__sample_weight=w)
    else:
        m.fit(X, y, sample_weight=w)
    return m


class Estrategia:
    """Una receta completa: qué entradas, qué objetivo, con qué datos entrenar y si se acota la salida."""

    def __init__(self, nombre, modelo, anio=False, rel_in=False, rel_y=False, ventana=None, vida_media=None, clip=None, sin_sumas=False):
        self.sin_sumas = sin_sumas
        self.nombre, self.modelo, self.anio, self.rel_in, self.rel_y = nombre, modelo, anio, rel_in, rel_y
        self.ventana, self.vida_media, self.clip = ventana, vida_media, clip

    def entradas(self, df, X0, ref=None, anio=None):
        Z = df[X0].drop(columns=SUMAS if self.sin_sumas else []).copy()
        r = df["ref"].values if ref is None else ref
        if self.rel_in:
            for c in SUMAS:
                Z[c] = Z[c] - r
        if self.anio:
            Z["anio"] = (df.temporada if anio is None else anio)
        return Z

    def entrenar(self, df, X0, hasta):
        d = df[df.temporada <= hasta]
        if self.ventana:
            d = d[d.temporada > hasta - self.ventana]
        w = None
        if self.vida_media:
            w = 0.5 ** ((hasta - d.temporada.values) / self.vida_media)
        y = d.puntos_totales - d.ref if self.rel_y else d.puntos_totales
        self.m = ajustar(crear(self.modelo), self.entradas(d, X0), y, w)
        self.X0 = X0
        return self

    def predecir(self, df, ref=None, anio=None):
        p = self.m.predict(self.entradas(df, self.X0, ref, anio))
        r = df["ref"].values if ref is None else ref
        if self.rel_y:
            p = p + r
        return np.clip(p, *self.clip) if self.clip else p


# ---------------------------------------------------------------- 1. drift
def medir_drift(ds, X0, carpeta):
    tr, rec = ds[ds.temporada <= FIN_TRAIN], ds[ds.temporada > PRUEBA]
    filas = []
    for c in X0 + ["puntos_totales"]:
        filas.append({"variable": c, "media_train": tr[c].mean(), "media_reciente": rec[c].mean(),
                      "PSI": psi(tr[c], rec[c]), "KS": ks_2samp(tr[c].dropna(), rec[c].dropna()).statistic})
    # las mismas variables en versión relativa (resta de la media de liga reciente)
    for c in SUMAS:
        a, b = tr[c] - tr.ref, rec[c] - rec.ref
        filas.append({"variable": c + "  [relativa a la liga]", "media_train": a.mean(), "media_reciente": b.mean(),
                      "PSI": psi(a, b), "KS": ks_2samp(a.dropna(), b.dropna()).statistic})
    a, b = tr.puntos_totales - tr.ref, rec.puntos_totales - rec.ref
    filas.append({"variable": "puntos_totales  [relativo a la liga]", "media_train": a.mean(), "media_reciente": b.mean(),
                  "PSI": psi(a, b), "KS": ks_2samp(a, b).statistic})
    t = pd.DataFrame(filas).round(3)
    t["veredicto"] = np.where(t.PSI > .25, "DRIFT", np.where(t.PSI > .1, "vigilar", "estable"))
    t.to_csv(carpeta / "drift_entradas_y_objetivo.csv", index=False)

    por_t = ds.groupby("temporada").agg(partidos=("game_id", "size"), media_puntos=("puntos_totales", "mean"),
                                        desv_puntos=("puntos_totales", "std"))
    por_t["PSI_objetivo_vs_train"] = [psi(tr.puntos_totales, ds[ds.temporada == s].puntos_totales) for s in por_t.index]
    por_t["PSI_objetivo_relativo"] = [psi(tr.puntos_totales - tr.ref, (ds[ds.temporada == s].puntos_totales - ds[ds.temporada == s].ref))
                                      for s in por_t.index]
    por_t.round(3).to_csv(carpeta / "drift_por_temporada.csv")
    return t, por_t


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tabla", default="data/modelado_partidos.parquet")
    ap.add_argument("--mlflow", default="sqlite:///mlflow.db")
    ap.add_argument("--salida", default="docs/drift")
    a = ap.parse_args()
    carpeta = pathlib.Path(a.salida); carpeta.mkdir(parents=True, exist_ok=True)

    ds = pd.read_parquet(a.tabla).sort_values(["game_date", "game_id"]).reset_index(drop=True)
    X0 = [c for c in ds.columns if c not in IDENT + SALIDAS]
    ds["ref"] = media_liga_reciente(ds)
    tr, te, l2 = ds[ds.temporada <= FIN_TRAIN], ds[ds.temporada == PRUEBA], ds[ds.temporada > PRUEBA]
    print(f"{len(ds):,} partidos | train {len(tr):,} (media {tr.puntos_totales.mean():.1f}) | prueba {len(te):,} | "
          f"lote 2 {len(l2):,} (media {l2.puntos_totales.mean():.1f})")

    # ---- 1. drift
    t, por_t = medir_drift(ds, X0, carpeta)
    pd.set_option("display.width", 200)
    print("\n=== DRIFT (train <=2022 vs temporadas >=2024) ===\n", t.to_string(index=False))

    # rango realista para el clip: percentiles 0,1% y 99,9% de todo lo visto, con 10 % de margen
    lo, hi = np.quantile(ds.puntos_totales, [.001, .999])
    CLIP = (float(np.floor(lo * .9)), float(np.ceil(hi * 1.1)))
    print(f"\nRango realista (p0.1 y p99.9 de todo el histórico, con 10 % de margen): {CLIP}")

    # ---- 2. estrategias
    E = [Estrategia("A_actual_ridge", "ridge"), Estrategia("A_actual_hgb", "hgb"),
         Estrategia("B_con_anio_ridge", "ridge", anio=True), Estrategia("B_con_anio_hgb", "hgb", anio=True),
         Estrategia("B2_solo_anio_ridge", "ridge", anio=True, sin_sumas=True), Estrategia("B2_solo_anio_hgb", "hgb", anio=True, sin_sumas=True),
         Estrategia("C_entradas_relativas_ridge", "ridge", rel_in=True), Estrategia("C_entradas_relativas_hgb", "hgb", rel_in=True),
         Estrategia("D_objetivo_relativo_ridge", "ridge", rel_in=True, rel_y=True),
         Estrategia("D_objetivo_relativo_hgb", "hgb", rel_in=True, rel_y=True),
         Estrategia("E_D_ventana8_ridge", "ridge", rel_in=True, rel_y=True, ventana=8),
         Estrategia("E_D_ventana8_hgb", "hgb", rel_in=True, rel_y=True, ventana=8),
         Estrategia("F_D_recencia_vm3_ridge", "ridge", rel_in=True, rel_y=True, vida_media=3),
         Estrategia("F_D_recencia_vm3_hgb", "hgb", rel_in=True, rel_y=True, vida_media=3),
         Estrategia("G_D_ridge_clip", "ridge", rel_in=True, rel_y=True, clip=CLIP),
         Estrategia("G_D_hgb_clip", "hgb", rel_in=True, rel_y=True, clip=CLIP)]
    mlflow.set_tracking_uri(a.mlflow)
    mlflow.set_experiment("hoopmind_obj2_drift")
    filas = []
    for e in E:
        e.entrenar(ds, X0, FIN_TRAIN)
        fila = {"estrategia": e.nombre}
        with mlflow.start_run(run_name=e.nombre):
            mlflow.log_params({"modelo": e.modelo, "usa_anio": e.anio, "entradas_relativas": e.rel_in,
                               "objetivo_relativo": e.rel_y, "ventana_temporadas": e.ventana, "vida_media": e.vida_media,
                               "clip": str(e.clip), "ventana_liga_partidos": VENT_LIGA, "entrena_hasta": FIN_TRAIN})
            for tag, d in [("prueba", te), ("lote2", l2)]:
                m = metr(d.puntos_totales, e.predecir(d))
                mlflow.log_metrics({f"{tag}_{k}": v for k, v in m.items()})
                fila.update({f"{tag}_{k}": v for k, v in m.items()})
            filas.append(fila)
    r1 = pd.DataFrame(filas).round(3)
    r1.to_csv(carpeta / "estrategias_validacion_temporal.csv", index=False)
    print("\n=== VALIDACIÓN TEMPORAL (entrena <=2022) ===\n", r1[["estrategia", "prueba_mae", "prueba_r2", "lote2_mae", "lote2_rmse", "lote2_r2", "lote2_sesgo"]].to_string(index=False))

    # ---- 2b. backtest anual (origen móvil): entrena con todo lo anterior, predice la temporada siguiente
    ests_bt = {"A_actual_hgb": lambda: Estrategia("A", "hgb"), "B_con_anio_ridge": lambda: Estrategia("B", "ridge", anio=True),
               "A_actual_ridge": lambda: Estrategia("A", "ridge"),
               "D_objetivo_relativo_ridge": lambda: Estrategia("D", "ridge", rel_in=True, rel_y=True),
               "D_objetivo_relativo_hgb": lambda: Estrategia("D", "hgb", rel_in=True, rel_y=True)}
    bt = []
    for s in range(2012, int(ds.temporada.max()) + 1):
        d = ds[ds.temporada == s]
        if len(d) < 150:
            continue
        for nombre, f in ests_bt.items():
            e = f().entrenar(ds, X0, s - 1)
            bt.append({"temporada": s, "estrategia": nombre, "mae": mean_absolute_error(d.puntos_totales, e.predecir(d)),
                       "sesgo": float(np.mean(e.predecir(d) - d.puntos_totales))})
    bt = pd.DataFrame(bt)
    bt_p = bt.pivot(index="temporada", columns="estrategia", values="mae").round(2)
    bt_p.to_csv(carpeta / "backtest_anual_mae.csv")
    print("\n=== BACKTEST ANUAL (MAE por temporada, reentrenando cada año) ===\n", bt_p.to_string())

    # ---- 4. modelo fijo vs reentrenado
    fijos = {"A_actual_hgb": Estrategia("A", "hgb").entrenar(ds, X0, 2012), "D_objetivo_relativo_ridge": Estrategia("D", "ridge", rel_in=True, rel_y=True).entrenar(ds, X0, 2012)}
    fj = []
    for s in range(2013, int(ds.temporada.max()) + 1):
        d = ds[ds.temporada == s]
        if len(d) < 150:
            continue
        for n, m in fijos.items():
            fj.append({"temporada": s, "estrategia": n + " (fijo, entrenado hasta 2012)", "mae": mean_absolute_error(d.puntos_totales, m.predecir(d))})
    fj = pd.concat([pd.DataFrame(fj), bt[bt.estrategia.isin(fijos)].assign(estrategia=lambda x: x.estrategia + " (reentrenado cada año)")[["temporada", "estrategia", "mae"]]])
    fj_p = fj.pivot(index="temporada", columns="estrategia", values="mae").round(2)
    fj_p.to_csv(carpeta / "modelo_fijo_vs_reentrenado_mae.csv")
    print("\n=== FIJO vs REENTRENADO (MAE) ===\n", fj_p.loc[[2013, 2016, 2019, 2022, 2024, 2025]].to_string())

    # ---- 3. prueba de estrés 2050
    base = l2.copy()
    pend = np.polyfit(por_t.index[(por_t.index >= 2010) & (por_t.partidos > 500)],
                      por_t.media_puntos[(por_t.index >= 2010) & (por_t.partidos > 500)], 1)[0]
    d2050 = pend * (2050 - 2025)
    print(f"\nTendencia 2010-2025: +{pend:.2f} puntos por temporada -> extrapolada a 2050: +{d2050:.0f} puntos sobre hoy")
    escenarios = {"1. año = 2050 (resto igual)": dict(anio=2050, delta=0.0),
                  f"2. liga 2050 siguiendo la tendencia (+{d2050:.0f})": dict(anio=2050, delta=d2050),
                  "3. liga loca (+800 pts, datos absurdos)": dict(anio=2050, delta=800.0)}
    sel = ["A_actual_ridge", "A_actual_hgb", "B_con_anio_ridge", "B2_solo_anio_ridge", "B2_solo_anio_hgb", "D_objetivo_relativo_ridge", "D_objetivo_relativo_hgb", "G_D_ridge_clip", "G_D_hgb_clip"]
    es = {e.nombre: e for e in E}
    est = []
    for esc, cfg in escenarios.items():
        dd = base.copy()
        dd["ref"] = dd["ref"] + cfg["delta"]                 # la media de liga reciente también sube (se calcula de datos reales)
        for c in SUMAS:
            dd[c] = dd[c] + cfg["delta"]                      # los resúmenes de puntos suben con la liga
        for n in sel:
            e = es[n]
            p = e.predecir(dd, anio=pd.Series(cfg["anio"], index=dd.index))
            est.append({"escenario": esc, "modelo": n, "pred_media": p.mean(), "pred_min": p.min(), "pred_max": p.max()})
    st = pd.DataFrame(est).round(1)
    st.to_csv(carpeta / "prueba_estres_2050.csv", index=False)
    print("\n=== PRUEBA DE ESTRÉS 2050 (predicción de puntos totales por partido) ===\n", st.to_string(index=False))
    with mlflow.start_run(run_name="estres_2050"):
        for _, r in st.iterrows():
            mlflow.log_metric(f"{r.modelo}__esc{r.escenario[0]}_pred_max", float(r.pred_max))
        mlflow.log_artifact(str(carpeta / "prueba_estres_2050.csv"))

    # ---- gráficos
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8), facecolor=CREMA)
    ax[0].plot(por_t.index, por_t.media_puntos, color=AZUL, marker="o")
    ax[0].axvline(FIN_TRAIN + .5, color=GRANATE, ls="--", lw=.8); ax[0].text(FIN_TRAIN + .7, 188, "lote 2\n(drift)", fontsize=8)
    ax[0].set_title("Puntos totales medios por partido"); ax[0].set_xlabel("temporada (año de inicio)")
    for n, c, ls in [("A_actual_hgb", GRANATE, "-"), ("B_con_anio_ridge", NARANJA, "-"), ("D_objetivo_relativo_ridge", AZUL, "-")]:
        ax[1].plot(bt_p.index, bt_p[n], color=c, marker="o", ms=3, ls=ls, label=n)
    ax[1].set_title("Error (MAE) por temporada, reentrenando"); ax[1].legend(fontsize=7)
    for n, c in [("A_actual_hgb (fijo, entrenado hasta 2012)", NARANJA), ("A_actual_hgb (reentrenado cada año)", AZUL)]:
        ax[2].plot(fj_p.index, fj_p[n], color=c, marker="o", ms=3, label=n.replace("A_actual_hgb ", ""))
    ax[2].set_title("Modelo fijo vs reentrenado (MAE)"); ax[2].legend(fontsize=7)
    for a_ in ax:
        a_.set_facecolor(CREMA); a_.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(carpeta / "drift_objetivo2.png", dpi=130); plt.close()
    mlflow.set_experiment("hoopmind_obj2_drift")
    with mlflow.start_run(run_name="resumen_drift"):
        mlflow.log_artifact(str(carpeta / "drift_entradas_y_objetivo.csv")); mlflow.log_artifact(str(carpeta / "drift_objetivo2.png"))
    print("\nListo. Resultados en", carpeta)


if __name__ == "__main__":
    main()
