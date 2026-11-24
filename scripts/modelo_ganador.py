"""modelo_ganador.py · HoopMind

Guardar y cargar el modelo del objetivo 1 (quién gana) como archivo, para que la API lo use
igual que usa modelo_puntos.joblib. Se guarda un diccionario con el modelo y la lista de columnas de entrada
(en ese orden), así que se puede cargar sin necesidad de tener este fichero en el PYTHONPATH.

Uso desde la API:
  from modelo_ganador import cargar_modelo_ganador, probabilidad_local
  p = probabilidad_local("models/modelo_ganador.joblib", {"dif_pts_favor_ult10": 3.2, ...})   # las 15 entradas
"""
import pathlib

import joblib
import pandas as pd


def guardar_modelo_ganador(modelo, columnas, ruta, nombre="regresion_logistica"):
    pathlib.Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"modelo": modelo, "columnas": list(columnas), "nombre": nombre}, ruta)
    return ruta


def cargar_modelo_ganador(ruta):
    """Devuelve el diccionario {'modelo', 'columnas', 'nombre'}."""
    return joblib.load(ruta)


def probabilidad_local(ruta, entradas):
    """Probabilidad de que gane el local. entradas: dict (una fila) o DataFrame con las columnas de entrada."""
    g = cargar_modelo_ganador(ruta)
    df = pd.DataFrame([entradas]) if isinstance(entradas, dict) else pd.DataFrame(entradas)
    faltan = [c for c in g["columnas"] if c not in df.columns]
    if faltan:
        raise ValueError(f"Faltan entradas: {faltan}")
    return g["modelo"].predict_proba(df[g["columnas"]])[:, 1]
