#!/usr/bin/env python3
"""validacion_temporal.py · HoopMind

Validación cruzada TEMPORAL con sklearn TimeSeriesSplit (5 cortes) para los dos objetivos.
Recomendada por el profe: en series de tiempo NO se mezclan filas al azar; cada corte entrena con el
pasado y valida con el bloque siguiente (nunca con el futuro hacia atrás).

  corte 1: entrena [####]                 valida [#]
  corte 2: entrena [#####]                valida  [#]
  ...
  corte 5: entrena [########]             valida      [#]

SOLO usa el conjunto de entrenamiento (temporadas <= --fin-train, 2000-01 a 2022-23).
La temporada de prueba (2023-24) y el lote 2 (2024-25 en adelante) NO se tocan: siguen siendo
el test final y el medidor de drift de modelos_base.py.

Objetivo 1 (home_win):       accuracy y ROC-AUC por corte (regresión logística, boosting y baseline "gana el local").
Objetivo 2 (puntos totales): MAE, RMSE y sesgo por corte, comparando *_relativo (objetivo relativo a la liga,
                             sin año, con tope) frente a *_absoluto (diseño antiguo, solo para comparar).
Además se guarda la temporada media de cada bloque de validación, para ver si el absoluto empeora en años recientes.

MLflow: un run por modelo con métricas por corte (paso = nº de corte, p. ej. cv_mae paso 1..5)
y también corte1_mae ... corte5_mae, más media y desviación (cv_mae_media, cv_mae_std).

Uso:
  python3 scripts/validacion_temporal.py --tabla data/modelado_partidos.parquet
  python3 scripts/validacion_temporal.py --sintetico          # datos inventados, solo para probar que funciona
  mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
import argparse
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from modelo_puntos import CLIP, ModeloPuntosRelativo, media_liga_reciente
from modelos_base import AZUL, CREMA, GRANATE, IDENT, SALIDAS
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier as HGBC, HistGradientBoostingRegressor as HGBR
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

N_CORTES = 5


def sintetico(n_temp=24, por_temp=1230, seed=0):
    """Tabla inventada con la MISMA forma que modelado_partidos.parquet (solo para probar el script)."""
    rng = np.random.default_rng(seed)
    filas = []
    for t in range(2000, 2000 + n_temp):
        nivel = 195 + (t - 2000) * 1.6                    # los puntos suben con los años (el drift real)
        fechas = pd.to_datetime(f"{t}-10-25") + pd.to_timedelta(np.sort(rng.integers(0, 170, por_temp)), unit="D")
        dif = rng.normal(0, 1, (por_temp, 13))
        fuerza = dif[:, 0] * 3 + dif[:, 1] * 2
        win = (fuerza + rng.normal(0, 6, por_temp) + 1.5 > 0).astype(int)
        pts = nivel + dif[:, 2] * 4 + rng.normal(0, 14, por_temp)
        d = pd.DataFrame(dif, columns=[f"dif_{i}" for i in range(13)])
        d["suma_pts_favor_ult10"] = nivel / 2 + rng.normal(0, 4, por_temp) + dif[:, 2] * 2
        d["suma_pts_contra_ult10"] = nivel / 2 + rng.normal(0, 4, por_temp) + dif[:, 2] * 2
        d["game_date"], d["temporada"], d["home_win"], d["puntos_totales"] = fechas, t, win, pts
        filas.append(d)
    ds = pd.concat(filas, ignore_index=True)
    ds["game_id"] = np.arange(len(ds))
    ds["home_team_id"], ds["away_team_id"] = 1, 2
    return ds


def pre(m):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), m)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tabla", default="data/modelado_partidos.parquet")
    ap.add_argument("--fin-train", type=int, default=2022)
    ap.add_argument("--mlflow", default="sqlite:///mlflow.db")
    ap.add_argument("--salida", default="docs/validacion_temporal", help="carpeta para CSV y figura")
    ap.add_argument("--sintetico", action="store_true", help="usa datos inventados (no hace falta la tabla real)")
    a = ap.parse_args()

    ds = sintetico() if a.sintetico else pd.read_parquet(a.tabla)
    ds = ds.sort_values(["game_date", "game_id"]).reset_index(drop=True)     # TimeSeriesSplit exige orden cronológico
    X = [c for c in ds.columns if c not in IDENT + SALIDAS]
    ds["ref_liga"] = media_liga_reciente(ds)                                 # solo mira partidos ANTERIORES: sin fuga
    # Solo entrenamiento: la prueba (2023-24) y el lote 2 quedan fuera de la validación cruzada.
    tv = ds[ds.temporada <= a.fin_train].reset_index(drop=True)
    cortes = list(TimeSeriesSplit(n_splits=N_CORTES).split(tv))
    print(f"{len(tv):,} partidos de entrenamiento (temporadas <= {a.fin_train}) | {len(X)} entradas | {N_CORTES} cortes")
    info = []
    for i, (itr, iva) in enumerate(cortes, 1):
        t_tr, t_va = tv.temporada.iloc[itr], tv.temporada.iloc[iva]
        info.append({"corte": i, "n_entrena": len(itr), "n_valida": len(iva),
                     "entrena_temporadas": f"{t_tr.min()}-{t_tr.max()}", "valida_temporadas": f"{t_va.min()}-{t_va.max()}",
                     "valida_temporada_media": round(float(t_va.mean()), 2)})
        print(f"  corte {i}: entrena {len(itr):>6,} ({t_tr.min()}-{t_tr.max()}) | valida {len(iva):>6,} ({t_va.min()}-{t_va.max()})")
    info = pd.DataFrame(info)
    salida = pathlib.Path(a.salida)
    salida.mkdir(parents=True, exist_ok=True)
    info.to_csv(salida / "cortes.csv", index=False)

    mlflow.set_tracking_uri(a.mlflow)
    filas = []

    def correr(exp, nombre, funcion, params=None):
        """Ejecuta un modelo en los 5 cortes y lo registra en MLflow. funcion(itr, iva) -> dict de métricas."""
        mlflow.set_experiment(exp)
        res = [funcion(itr, iva) for itr, iva in cortes]
        with mlflow.start_run(run_name=f"cv_{nombre}"):
            mlflow.log_params({"validacion": "TimeSeriesSplit", "n_cortes": N_CORTES, "entrena_hasta": a.fin_train,
                               "n_entradas": len(X), **(params or {})})
            for k in res[0]:
                v = np.array([r[k] for r in res])
                for i, x in enumerate(v, 1):
                    mlflow.log_metric(f"cv_{k}", float(x), step=i)       # curva por corte
                    mlflow.log_metric(f"corte{i}_{k}", float(x))         # y como número suelto
                mlflow.log_metrics({f"cv_{k}_media": float(v.mean()), f"cv_{k}_std": float(v.std(ddof=1))})
            mlflow.log_artifact(str(salida / "cortes.csv"))
        for i, r in enumerate(res, 1):
            filas.append({"experimento": exp, "modelo": nombre, "corte": i, **r})
        print(f"  {nombre:<22}" + "  ".join(f"{k}={np.mean([r[k] for r in res]):.3f}±{np.std([r[k] for r in res], ddof=1):.3f}" for k in res[0]))

    # ---------------- Objetivo 1: home_win ----------------
    print("\nObjetivo 1 · home_win")
    y = tv.home_win.values

    def clasif(make):
        def f(itr, iva):
            m = make().fit(tv.loc[itr, X], y[itr])
            p = m.predict_proba(tv.loc[iva, X])[:, 1]
            return {"accuracy": accuracy_score(y[iva], (p >= .5).astype(int)), "auc": roc_auc_score(y[iva], p)}
        return f

    def base_local(itr, iva):      # "gana siempre el local": acierta lo que valga la tasa de victorias locales del bloque
        m = DummyClassifier(strategy="constant", constant=1).fit(tv.loc[itr, X], y[itr])
        return {"accuracy": accuracy_score(y[iva], m.predict(tv.loc[iva, X])), "auc": 0.5}

    correr("hoopmind_obj1_home_win", "baseline_gana_local", base_local)
    correr("hoopmind_obj1_home_win", "regresion_logistica", clasif(lambda: pre(LogisticRegression(max_iter=2000))))
    correr("hoopmind_obj1_home_win", "boosting", clasif(lambda: HGBC(max_iter=200, learning_rate=.05, random_state=0)))

    # ---------------- Objetivo 2: puntos_totales ----------------
    print("\nObjetivo 2 · puntos_totales  (relativo = sin año + media de liga + tope; absoluto = diseño antiguo)")
    y = tv.puntos_totales.values
    ref = tv.ref_liga.values

    def metricas(real, p):
        return {"mae": mean_absolute_error(real, p), "rmse": float(np.sqrt(mean_squared_error(real, p))),
                "sesgo": float(np.mean(p - real))}

    def absoluto(make):
        def f(itr, iva):
            return metricas(y[iva], make().fit(tv.loc[itr, X], y[itr]).predict(tv.loc[iva, X]))
        return f

    def relativo(tipo):
        def f(itr, iva):
            m = ModeloPuntosRelativo(tipo).fit(tv.loc[itr, X], y[itr], ref[itr])
            return metricas(y[iva], m.predict(tv.loc[iva, X], ref[iva]))
        return f

    e2 = "hoopmind_obj2_puntos_totales"
    correr(e2, "baseline_media", absoluto(lambda: DummyRegressor(strategy="mean")))
    correr(e2, "ridge_absoluto", absoluto(lambda: pre(Ridge(alpha=10))), {"objetivo_relativo": False})
    correr(e2, "boosting_absoluto", absoluto(lambda: HGBR(max_iter=200, learning_rate=.05, random_state=0)), {"objetivo_relativo": False})
    correr(e2, "ridge_relativo", relativo("ridge"), {"objetivo_relativo": True, "clip": str(CLIP)})
    correr(e2, "boosting_relativo", relativo("boosting"), {"objetivo_relativo": True, "clip": str(CLIP)})

    res = pd.DataFrame(filas).merge(info[["corte", "valida_temporadas", "valida_temporada_media"]], on="corte")
    res.round(4).to_csv(salida / "metricas_por_corte.csv", index=False)
    resumen = (res.drop(columns=["corte", "valida_temporadas", "valida_temporada_media"])
               .groupby(["experimento", "modelo"]).agg(["mean", "std"]).round(3))
    resumen.columns = [f"{m}_{s}" for m, s in resumen.columns]
    resumen.dropna(axis=1, how="all").to_csv(salida / "resumen_cv.csv")

    # Figura: MAE y sesgo por corte, relativo frente a absoluto
    p2 = res[res.experimento == e2]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), facecolor=CREMA)
    estilo = {"ridge_absoluto": (GRANATE, "--"), "boosting_absoluto": ("#9a5b5b", "--"),
              "ridge_relativo": (AZUL, "-"), "boosting_relativo": ("#5b9bd5", "-")}
    for nombre, (col, ls) in estilo.items():
        d = p2[p2.modelo == nombre].sort_values("corte")
        for a_, k in zip(ax, ("mae", "sesgo")):
            a_.plot(d.valida_temporada_media, d[k], color=col, ls=ls, marker="o", label=nombre)
    ax[0].set_title("Puntos totales: MAE por corte"); ax[1].set_title("Puntos totales: sesgo (predicho - real) por corte")
    ax[1].axhline(0, color="k", lw=.7)
    for a_ in ax:
        a_.set_xlabel("temporada media del bloque de validación"); a_.set_facecolor(CREMA)
        a_.spines[["top", "right"]].set_visible(False)
    ax[0].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(salida / "relativo_vs_absoluto_por_corte.png", dpi=130); plt.close()

    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    print("\nPuntos totales por corte (sesgo: negativo = el modelo se queda corto):")
    print(p2.pivot(index="modelo", columns="corte", values="sesgo").round(2).to_string())
    print(f"\nArchivos en {salida}/")


if __name__ == "__main__":
    main()
