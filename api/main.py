"""HoopMind · API FastAPI

Dos modelos, dos endpoints:
  POST /predict/ganador  -> probabilidad de que gane el local (modelo models/modelo_ganador.joblib)
  POST /predict/puntos   -> puntos totales del partido (modelo models/modelo_puntos.joblib, relativo a la liga)
  GET  /equipos          -> equipos disponibles y fecha de los datos (models/estado_equipos.json)
  POST /predict/partido  -> elige local y visitante: la API calcula las 15 entradas y predice ganador y puntos
  GET  /health           -> comprueba que la API y los modelos están listos
  GET  /                 -> página sencilla para probar a mano
  GET  /docs             -> pantalla de pruebas automática de FastAPI

Los modelos los crea scripts/modelos_base.py. La carpeta se puede cambiar con la variable HOOPMIND_MODELOS.

Arrancar (desde la raíz del repo):
  pip install -r api/requirements.txt
  uvicorn api.main:app --reload
"""
import json
import math
import os
import pathlib
import sys

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

RAIZ = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "scripts"))   # el .joblib de puntos guarda una clase que vive en scripts/
from modelo_ganador import cargar_modelo_ganador  # noqa: E402
from modelo_puntos import CLIP, cargar_modelo  # noqa: E402

LIGA_POR_DEFECTO = 229.3   # media de puntos totales del lote 2 (2024-25 en adelante); sustituir por la media de los últimos 600 partidos

app = FastAPI(title="HoopMind API", version="1.0",
              description="Predice el ganador de un partido NBA y sus puntos totales a partir de la forma reciente "
                          "de los dos equipos (solo datos de equipo y partido, sin jugadores).")


def _d(texto):
    return Field(..., description=texto)


class Entradas(BaseModel):
    """Las 15 entradas de un partido, calculadas ANTES del partido. 'dif_' = local - visitante."""
    dif_pts_favor_ult10: float = _d("Puntos a favor (media de los últimos 10), local − visitante")
    dif_pts_contra_ult10: float = _d("Puntos en contra (últimos 10), local − visitante")
    dif_win_pct_ult10: float = _d("% de victorias en los últimos 10 (0 a 1), local − visitante")
    dif_tiro_campo_pct_ult10: float = _d("% de tiro de campo (0 a 1), local − visitante")
    dif_triples_pct_ult10: float = _d("% de triples (0 a 1), local − visitante")
    dif_libres_pct_ult10: float = _d("% de tiros libres (0 a 1), local − visitante")
    dif_rebotes_ult10: float = _d("Rebotes, local − visitante")
    dif_asistencias_ult10: float = _d("Asistencias, local − visitante")
    dif_perdidas_ult10: float = _d("Pérdidas, local − visitante")
    dif_robos_ult10: float = _d("Robos, local − visitante")
    dif_tapones_ult10: float = _d("Tapones, local − visitante")
    dif_dias_descanso: float = _d("Días de descanso (tope 7), local − visitante")
    dif_win_pct_temporada: float = _d("% de victorias en la temporada hasta hoy (0 a 1), local − visitante")
    suma_pts_favor_ult10: float = _d("Puntos a favor (últimos 10), local + visitante")
    suma_pts_contra_ult10: float = _d("Puntos en contra (últimos 10), local + visitante")


COLUMNAS = list(Entradas.model_fields)


class EntradasPuntos(Entradas):
    media_liga_reciente: float = Field(LIGA_POR_DEFECTO, ge=100, le=400,
                                       description="Media de puntos totales de los últimos 600 partidos jugados en la liga "
                                                   "(el modelo predice la desviación respecto a este número)")


_cache = {}


def _ruta(nombre):
    return pathlib.Path(os.environ.get("HOOPMIND_MODELOS", RAIZ / "models")) / nombre


def _cargar(clave, nombre, cargador):
    ruta = _ruta(nombre)
    if clave not in _cache:
        if not ruta.exists():
            raise HTTPException(503, f"Falta {ruta}. Genera los modelos con: python scripts/modelos_base.py --tabla data/modelado_partidos.parquet")
        _cache[clave] = cargador(ruta)
    return _cache[clave]


def _fila(entradas):
    import pandas as pd
    return pd.DataFrame([entradas.model_dump()])[COLUMNAS]


@app.get("/health")
def health():
    return {"estado": "ok", "modelo_ganador": _ruta("modelo_ganador.joblib").exists(),
            "modelo_puntos": _ruta("modelo_puntos.joblib").exists()}


@app.get("/contrato")
def contrato():
    """Lista de las 15 entradas, en orden."""
    return {"entradas": COLUMNAS, "media_liga_por_defecto": LIGA_POR_DEFECTO}


def _pred_ganador(df):
    g = _cargar("ganador", "modelo_ganador.joblib", cargar_modelo_ganador)
    p = float(g["modelo"].predict_proba(df[g["columnas"]])[:, 1][0])
    return {"prob_local": round(p, 4), "prob_visitante": round(1 - p, 4),
            "ganador": "local" if p >= .5 else "visitante", "modelo": g["nombre"]}


def _pred_puntos(df, ref):
    m = _cargar("puntos", "modelo_puntos.joblib", cargar_modelo)
    p = float(m.predict(df, ref)[0])
    return {"puntos_totales": round(p, 1), "media_liga_usada": ref, "rango_permitido": list(CLIP), "modelo": m.tipo + "_relativo"}


@app.post("/predict/ganador")
def predecir_ganador(e: Entradas):
    return _pred_ganador(_fila(e))


@app.post("/predict/puntos")
def predecir_puntos(e: EntradasPuntos):
    return _pred_puntos(_fila(e), e.media_liga_reciente)


# ---------- Elegir dos equipos ----------
def _estado():
    ruta = _ruta("estado_equipos.json")
    if "estado" not in _cache:
        if not ruta.exists():
            raise HTTPException(503, f"Falta {ruta}. Genera el estado de los equipos con: python scripts/estado_equipos.py --curated data/curated_export")
        _cache["estado"] = json.loads(ruta.read_text(encoding="utf-8"))
    return _cache["estado"]


class Partido(BaseModel):
    local_id: int = Field(..., description="id del equipo local (ver /equipos)")
    visitante_id: int = Field(..., description="id del equipo visitante (ver /equipos)")
    dias_descanso_local: float = Field(2, ge=0, le=7, description="Días desde su último partido (tope 7)")
    dias_descanso_visitante: float = Field(2, ge=0, le=7, description="Días desde su último partido (tope 7)")


def _v(x):
    return float("nan") if x is None else x


def _limpio(d):
    return {k: (None if isinstance(v, float) and math.isnan(v) else round(v, 4)) for k, v in d.items()}


@app.get("/equipos")
def equipos():
    est = _estado()
    return {"datos_hasta": est["fecha_corte"], "temporada": est["temporada"], "media_liga_reciente": est["media_liga_reciente"],
            "equipos": [{"id": x["id"], "nombre": x["nombre"], "ultimo_partido": x["ultimo_partido"],
                         "partidos_temporada": x["partidos_temporada"],
                         "forma": {k: v for k, v in x.items() if k.endswith("_ult10") or k == "win_pct_temporada"}}
                        for x in est["equipos"]]}


@app.post("/predict/partido")
def predecir_partido(p: Partido):
    est = _estado()
    por_id = {x["id"]: x for x in est["equipos"]}
    if p.local_id == p.visitante_id:
        raise HTTPException(422, "El local y el visitante tienen que ser equipos distintos")
    for i in (p.local_id, p.visitante_id):
        if i not in por_id:
            raise HTTPException(404, f"Equipo {i} no encontrado; consulta /equipos")
    lo, vi = por_id[p.local_id], por_id[p.visitante_id]
    ent = {}
    for col in COLUMNAS:
        if col.startswith("dif_") and col.endswith("_ult10"):
            ent[col] = _v(lo.get(col[4:])) - _v(vi.get(col[4:]))
        elif col.startswith("suma_"):
            ent[col] = _v(lo.get(col[5:])) + _v(vi.get(col[5:]))
    ent["dif_dias_descanso"] = p.dias_descanso_local - p.dias_descanso_visitante
    ent["dif_win_pct_temporada"] = _v(lo["win_pct_temporada"]) - _v(vi["win_pct_temporada"])
    import pandas as pd
    df = pd.DataFrame([ent])[COLUMNAS]
    ref = est["media_liga_reciente"]
    return {"local": lo["nombre"], "visitante": vi["nombre"], "datos_hasta": est["fecha_corte"],
            "entradas": _limpio(ent), "ganador": _pred_ganador(df), "puntos": _pred_puntos(df, ref)}


app.mount("/static", StaticFiles(directory=RAIZ / "api" / "static"), name="static")


@app.get("/", include_in_schema=False)
def inicio():
    return FileResponse(RAIZ / "api" / "static" / "index.html")
