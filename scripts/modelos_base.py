#!/usr/bin/env python3
"""modelos_base.py · HoopMind

EDA rápido y modelos BASE de los dos objetivos, con seguimiento en MLflow.
Entrada: la tabla compacta de modelado (data/modelado_partidos.parquet, de construir_modelado.py).

Cortes por fecha (no se mezclan filas al azar):
  entrenamiento  temporadas <= 2022   (2000-01 a 2022-23)
  prueba         temporada  == 2023   (2023-24)
  lote 2         temporadas >= 2024   (se usa solo para MEDIR cuánto empeora el modelo: drift)
El corte es orientativo (el plan lo deja pendiente): se cambia con --fin-train y --prueba.

Objetivo 1 (clasificación): home_win       -> baseline "gana siempre el local", regresión logística, boosting
Objetivo 2 (regresión):     puntos_totales -> baseline "la media", Ridge y boosting RELATIVOS a la media de liga
                            (arreglo del drift: sin año, se predice puntos - media_liga y se vuelve a sumar, tope 127-319;
                            los modelos *_absoluto se conservan solo para comparar). Ver scripts/modelo_puntos.py

Uso:
  python3 scripts/modelos_base.py --tabla data/modelado_partidos.parquet
  mlflow ui --backend-store-uri sqlite:///mlflow.db        # y abrir http://127.0.0.1:5000
"""
import argparse
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from modelo_ganador import guardar_modelo_ganador
from modelo_puntos import CLIP, VENT_LIGA, ModeloPuntosRelativo, guardar_modelo, media_liga_reciente
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier as HGBC, HistGradientBoostingRegressor as HGBR
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

IDENT = ["game_id", "game_date", "temporada", "home_team_id", "away_team_id"]
SALIDAS = ["home_win", "puntos_totales"]
AZUL, GRANATE, CREMA = "#174D75", "#21060A", "#F8F5E4"


def eda(ds, X, carpeta):
    carpeta.mkdir(parents=True, exist_ok=True)
    por_t = ds.groupby("temporada").agg(partidos=("game_id", "size"), pct_local=("home_win", "mean"),
                                        puntos=("puntos_totales", "mean"))
    por_t.round(3).to_csv(carpeta / "resumen_por_temporada.csv")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4), facecolor=CREMA)
    ax[0].plot(por_t.index, por_t.pct_local * 100, color=AZUL, marker="o")
    ax[0].axhline(50, color=GRANATE, lw=.8, ls="--")
    ax[0].set_title("% de partidos que gana el local"); ax[0].set_xlabel("temporada (año de inicio)")
    ax[1].plot(por_t.index, por_t.puntos, color=AZUL, marker="o")
    ax[1].set_title("Puntos totales medios por partido"); ax[1].set_xlabel("temporada (año de inicio)")
    for a in ax:
        a.set_facecolor(CREMA); a.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(carpeta / "salidas_por_temporada.png", dpi=130); plt.close()

    corr = ds[X + SALIDAS].corr()
    fig, ax = plt.subplots(figsize=(8, 6.5), facecolor=CREMA)
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns, fontsize=7)
    plt.colorbar(im, shrink=.7); ax.set_title("Correlación entre entradas y salidas")
    plt.tight_layout(); plt.savefig(carpeta / "correlaciones.png", dpi=130); plt.close()
    c_out = corr.loc[X, SALIDAS].round(3)
    c_out.to_csv(carpeta / "correlacion_con_salidas.csv")
    print("Entradas más ligadas a home_win:\n", c_out["home_win"].abs().sort_values(ascending=False).head(5).round(3).to_string())
    print("Entradas más ligadas a puntos_totales:\n", c_out["puntos_totales"].abs().sort_values(ascending=False).head(5).round(3).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tabla", default="data/modelado_partidos.parquet")
    ap.add_argument("--fin-train", type=int, default=2022)
    ap.add_argument("--prueba", type=int, default=2023)
    ap.add_argument("--mlflow", default="sqlite:///mlflow.db")
    ap.add_argument("--eda", default="docs/eda")
    ap.add_argument("--modelos", default="models", help="carpeta donde se guarda modelo_puntos.joblib para la API")
    a = ap.parse_args()

    ds = pd.read_parquet(a.tabla).sort_values(["game_date", "game_id"]).reset_index(drop=True)
    X = [c for c in ds.columns if c not in IDENT + SALIDAS]
    ds["ref_liga"] = media_liga_reciente(ds)      # media de los 600 partidos anteriores (no es entrada; se suma al predecir)
    tr = ds.temporada <= a.fin_train
    te = ds.temporada == a.prueba
    l2 = ds.temporada > a.prueba
    print(f"{len(ds):,} partidos | {len(X)} entradas | entrenamiento {tr.sum():,} | prueba {te.sum():,} | lote 2 {l2.sum():,}")
    eda(ds, X, pathlib.Path(a.eda))

    mlflow.set_tracking_uri(a.mlflow)
    pre = lambda m: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), m)

    # ---------------- Objetivo 1: home_win ----------------
    mlflow.set_experiment("hoopmind_obj1_home_win")
    y = ds.home_win
    modelos = {"baseline_gana_local": DummyClassifier(strategy="constant", constant=1),
               "regresion_logistica": pre(LogisticRegression(max_iter=2000)),
               "boosting": HGBC(max_iter=200, learning_rate=.05, random_state=0)}
    for nombre, m in modelos.items():
        m.fit(ds.loc[tr, X], y[tr])
        with mlflow.start_run(run_name=nombre):
            mlflow.log_params({"n_entradas": len(X), "entrena_hasta": a.fin_train, "prueba": a.prueba})
            for tag, mask in [("prueba", te), ("lote2", l2)]:
                p = m.predict_proba(ds.loc[mask, X])[:, 1] if nombre != "baseline_gana_local" else np.full(mask.sum(), y[tr].mean())
                mlflow.log_metrics({f"{tag}_accuracy": accuracy_score(y[mask], m.predict(ds.loc[mask, X])),
                                    f"{tag}_auc": roc_auc_score(y[mask], p) if nombre != "baseline_gana_local" else 0.5,
                                    f"{tag}_logloss": log_loss(y[mask], np.clip(p, 1e-6, 1 - 1e-6)),
                                    f"{tag}_prob_media_local": float(p.mean()), f"{tag}_tasa_real_local": float(y[mask].mean())})
            mlflow.sklearn.log_model(m, name="modelo", serialization_format="cloudpickle")
            if nombre == "regresion_logistica":
                # El elegido para la API (el mejor en AUC): archivo con el modelo y la lista de columnas de entrada
                ruta = guardar_modelo_ganador(m, X, f"{a.modelos}/modelo_ganador.joblib", nombre)
                mlflow.log_artifact(ruta)

    # ---------------- Objetivo 2: puntos_totales (a prueba de drift) ----------------
    # Los *_absoluto son el diseño antiguo (solo para comparar). Los *_relativo son el modelo bueno:
    # sin año, objetivo = puntos - media de liga reciente, se vuelve a sumar al predecir y se acota a CLIP.
    # Detalle y explicación: scripts/modelo_puntos.py y docs/drift/EXPLICACION_DRIFT.md
    mlflow.set_experiment("hoopmind_obj2_puntos_totales")
    y = ds.puntos_totales
    ref = ds.ref_liga.values
    modelos = {"baseline_media": ("abs", DummyRegressor(strategy="mean")),
               "ridge_absoluto": ("abs", pre(Ridge(alpha=10))),
               "boosting_absoluto": ("abs", HGBR(max_iter=200, learning_rate=.05, random_state=0)),
               "ridge_relativo": ("rel", ModeloPuntosRelativo("ridge")),
               "boosting_relativo": ("rel", ModeloPuntosRelativo("boosting"))}
    for nombre, (modo, m) in modelos.items():
        if modo == "rel":
            m.fit(ds.loc[tr, X], y[tr], ref[tr])
        else:
            m.fit(ds.loc[tr, X], y[tr])
        with mlflow.start_run(run_name=nombre):
            mlflow.log_params({"n_entradas": len(X), "entrena_hasta": a.fin_train, "prueba": a.prueba,
                               "usa_anio": False, "objetivo_relativo": modo == "rel",
                               "ventana_liga_partidos": VENT_LIGA if modo == "rel" else None,
                               "clip": str(CLIP) if modo == "rel" else None})
            for tag, mask in [("prueba", te), ("lote2", l2)]:
                p = m.predict(ds.loc[mask, X], ref[mask]) if modo == "rel" else m.predict(ds.loc[mask, X])
                mlflow.log_metrics({f"{tag}_mae": mean_absolute_error(y[mask], p),
                                    f"{tag}_rmse": float(np.sqrt(mean_squared_error(y[mask], p))),
                                    f"{tag}_r2": r2_score(y[mask], p),
                                    f"{tag}_sesgo": float(np.mean(p - y[mask])),
                                    f"{tag}_media_predicha": float(p.mean()), f"{tag}_media_real": float(y[mask].mean())})
            if modo == "rel":
                # Modelo para la API: fichero .joblib (se carga con modelo_puntos.cargar_modelo) + copia en MLflow
                ruta = guardar_modelo(m, f"{a.modelos}/modelo_puntos_{nombre}.joblib")
                mlflow.log_artifact(ruta)
                if nombre == "ridge_relativo":
                    guardar_modelo(m, f"{a.modelos}/modelo_puntos.joblib")   # el elegido (más simple, extrapola bien)
                    mlflow.log_artifact(f"{a.modelos}/modelo_puntos.joblib")
            else:
                mlflow.sklearn.log_model(m, name="modelo", serialization_format="cloudpickle")

    # Resumen legible
    filas = []
    for exp in ("hoopmind_obj1_home_win", "hoopmind_obj2_puntos_totales"):
        for r in mlflow.search_runs(experiment_names=[exp]).to_dict("records"):
            filas.append({"experimento": exp, "modelo": r["tags.mlflow.runName"],
                          **{k.replace("metrics.", ""): round(v, 3) for k, v in r.items() if k.startswith("metrics.") and v == v}})
    res = pd.DataFrame(filas).sort_values(["experimento", "modelo"])
    pathlib.Path("docs").mkdir(exist_ok=True)
    res.to_csv("docs/resultados_modelos_base.csv", index=False)
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
