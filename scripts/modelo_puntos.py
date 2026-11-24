#!/usr/bin/env python3
"""modelo_puntos.py · HoopMind

Modelo del objetivo 2 (puntos totales de un partido) a prueba de drift.
Lo usan modelos_base.py (entrenar) y la futura API FastAPI (cargar y predecir).

Idea, en sencillo: la liga anota cada vez más (203,7 puntos de media en el entrenamiento, 229,3 desde 2024).
Si el modelo aprende "puntos absolutos" se queda anticuado. Así que:
  1. NO usa el año como entrada.
  2. Aprende a predecir CUÁNTO SE DESVÍA el partido de la media de liga reciente (puntos - media_liga).
  3. Al predecir, vuelve a SUMAR la media de liga reciente actual.
  4. Acota el resultado a un rango realista [127, 319] (red de seguridad contra datos corruptos).

  puntos_predichos = clip( media_liga_reciente + modelo(entradas relativas) , 127, 319 )

Uso desde la API (ver al final del fichero, función predecir_desde_ruta):
  from modelo_puntos import cargar_modelo, media_liga_actual
  modelo = cargar_modelo("models/modelo_puntos.joblib")
  ref = media_liga_actual(historico_puntos_totales)      # media de los últimos 600 partidos jugados
  puntos = modelo.predict(filas_de_entradas, ref)

Importante: el .joblib guarda una clase de este fichero, así que scripts/ debe estar en el PYTHONPATH
al cargarlo (sys.path.insert(0, "scripts") o copiar este fichero junto a la API).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor as HGBR
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Únicas entradas con nivel ABSOLUTO de puntos (las otras 13 son diferencias local-visitante, ya estables)
SUMAS = ["suma_pts_favor_ult10", "suma_pts_contra_ult10"]
VENT_LIGA = 600                  # partidos previos que forman la "media de liga reciente" (~ mes y medio)
CLIP = (127.0, 319.0)            # percentiles 0,1 % y 99,9 % del histórico con 10 % de margen (docs/drift)


def media_liga_reciente(ds, n=VENT_LIGA):
    """Para cada partido: media de puntos totales de los n partidos ANTERIORES por fecha.
    Sin fuga: el propio día no cuenta. El primer día usa el siguiente valor disponible.
    ds necesita las columnas game_date y puntos_totales. Devuelve un array alineado con ds."""
    d = ds.groupby("game_date").puntos_totales.agg(["size", "sum"]).sort_index()
    cn, cs = d["size"].cumsum().values, d["sum"].cumsum().values
    prev_n = np.r_[0, cn[:-1]]
    prev_s = np.r_[0, cs[:-1]]
    ref = np.empty(len(d))
    for i in range(len(d)):
        if prev_n[i] == 0:
            ref[i] = np.nan
            continue
        ini = np.searchsorted(prev_n, max(prev_n[i] - n, 0), side="left")
        ref[i] = (prev_s[i] - prev_s[ini]) / max(prev_n[i] - prev_n[ini], 1)
    r = pd.Series(ref, index=d.index).bfill()
    return ds.game_date.map(r).values


def media_liga_actual(puntos_totales_historico, n=VENT_LIGA):
    """Valor de 'media de liga reciente' para predecir un partido FUTURO: media de los últimos n partidos jugados.
    Recibe la serie de puntos totales ordenada de más antiguo a más reciente."""
    p = np.asarray(puntos_totales_historico, dtype=float)
    return float(p[-n:].mean())


def _crear(tipo):
    if tipo == "ridge":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10))
    if tipo == "boosting":
        return HGBR(max_iter=200, learning_rate=.05, random_state=0)
    raise ValueError(f"tipo desconocido: {tipo} (usa 'ridge' o 'boosting')")


class ModeloPuntosRelativo:
    """Predice puntos totales como media_liga + desviación aprendida. No usa el año."""

    def __init__(self, tipo="ridge", clip=CLIP):
        self.tipo, self.clip = tipo, tuple(clip)
        self.m = None
        self.columnas = None

    def _relativas(self, X, ref):
        Z = X[self.columnas].copy()
        ref = np.asarray(ref, dtype=float)
        for c in SUMAS:
            if c in Z.columns:
                Z[c] = Z[c] - ref
        return Z

    def fit(self, X, y, ref):
        """X: entradas (DataFrame, sin año). y: puntos_totales. ref: media de liga reciente de cada partido."""
        self.columnas = list(X.columns)
        ref = np.asarray(ref, dtype=float)
        self.m = _crear(self.tipo).fit(self._relativas(X, ref), np.asarray(y, dtype=float) - ref)
        return self

    def predict(self, X, ref):
        """ref: un número (liga actual) o un array con un valor por fila. Devuelve puntos totales ya 'deshechos'."""
        if self.m is None:
            raise RuntimeError("Modelo sin entrenar")
        ref = np.broadcast_to(np.asarray(ref, dtype=float), (len(X),))
        p = self.m.predict(self._relativas(X, ref)) + ref
        return np.clip(p, *self.clip)


def guardar_modelo(modelo, ruta):
    import pathlib
    import joblib
    pathlib.Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, ruta)
    return ruta


def cargar_modelo(ruta):
    import joblib
    return joblib.load(ruta)


def predecir_desde_ruta(ruta_modelo, entradas, ref):
    """Atajo para la API: carga el modelo y predice. entradas = DataFrame (o dict de listas) con las columnas de entrada."""
    modelo = cargar_modelo(ruta_modelo)
    return modelo.predict(pd.DataFrame(entradas), ref)
