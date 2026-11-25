"""HoopMind · API FastAPI

Dos modelos, dos endpoints:
  POST /predict/ganador  -> probabilidad de que gane el local (modelo models/modelo_ganador.joblib)
  POST /predict/puntos   -> puntos totales del partido (modelo models/modelo_puntos.joblib, relativo a la liga)
  GET  /equipos          -> equipos disponibles y fecha de los datos (models/estado_equipos.json)
  POST /predict/fecha    -> igual que /predict/partido pero en una fecha: pasada (forma con los 10 partidos anteriores y resultado real) o futura (estimación)
  GET  /serie-puntos     -> puntos medios por temporada (para el gráfico de drift de la web)
  POST /predict/partido  -> elige local y visitante: la API calcula las 15 entradas y predice ganador y puntos
  GET  /health           -> comprueba que la API y los modelos están listos
  GET  /                 -> página sencilla para probar a mano
  GET  /docs             -> pantalla de pruebas automática de FastAPI

Los modelos los crea scripts/modelos_base.py. La carpeta se puede cambiar con la variable HOOPMIND_MODELOS.

Arrancar (desde la raíz del repo):
  pip install -r api/requirements.txt
  uvicorn api.main:app --reload
"""
import csv
import datetime
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
    ent = _entradas(lo, vi, p.dias_descanso_local, p.dias_descanso_visitante)
    df = _df(ent)
    ref = est["media_liga_reciente"]
    return {"local": lo["nombre"], "visitante": vi["nombre"], "datos_hasta": est["fecha_corte"], "temporada": est["temporada"],
            "entradas": _limpio(ent), "ganador": _pred_ganador(df), "puntos": _pred_puntos(df, ref)}


def _entradas(lo, vi, dias_local, dias_visit):
    """Las 15 entradas a partir de la forma de cada equipo (mismas definiciones que construir_modelado.py)."""
    ent = {}
    for col in COLUMNAS:
        if col.startswith("dif_") and col.endswith("_ult10"):
            ent[col] = _v(lo.get(col[4:])) - _v(vi.get(col[4:]))
        elif col.startswith("suma_"):
            ent[col] = _v(lo.get(col[5:])) + _v(vi.get(col[5:]))
    ent["dif_dias_descanso"] = dias_local - dias_visit
    ent["dif_win_pct_temporada"] = _v(lo["win_pct_temporada"]) - _v(vi["win_pct_temporada"])
    return ent


def _df(ent):
    import pandas as pd
    return pd.DataFrame([ent])[COLUMNAS]


# ---------- Elegir equipos Y una fecha (pasada o futura) ----------
class PartidoFecha(BaseModel):
    local_id: int = Field(..., description="id del equipo local (ver /equipos)")
    visitante_id: int = Field(..., description="id del equipo visitante (ver /equipos)")
    fecha: datetime.date = Field(..., description="Fecha del partido (AAAA-MM-DD). Pasada: se usan sus 10 partidos anteriores a esa fecha. Futura: se usa el estado más reciente.")
    con_tendencia: bool = Field(True, description="Solo fechas futuras: si es true, sube el nivel de puntos de la liga siguiendo la tendencia de los últimos años")


def _historial():
    if "historial" not in _cache:
        ruta = _ruta("historial_equipos.csv.gz")
        if not ruta.exists():
            raise HTTPException(503, f"Falta {ruta}. Genera el histórico con: python scripts/estado_equipos.py --curated data/curated_export")
        import pandas as pd
        _cache["historial"] = pd.read_csv(ruta, parse_dates=["game_date"])
    return _cache["historial"]


ANOS_TENDENCIA = 10    # temporadas completas usadas para medir cuánto sube la liga por año
MAX_ANOS_EXTRAPOLAR = 10   # más allá, el nivel de puntos se queda fijo (no inventamos siglos de tendencia)


def _serie():
    ruta = RAIZ / "docs" / "drift" / "drift_por_temporada.csv"
    if not ruta.exists():
        raise HTTPException(404, f"Falta {ruta}")
    with open(ruta, encoding="utf-8") as f:
        return [{"temporada": int(r["temporada"]), "media": float(r["media_puntos"]), "partidos": int(r["partidos"])} for r in csv.DictReader(f)]


def _pendiente():
    """Puntos totales que sube la liga por año: recta (mínimos cuadrados) sobre las últimas 10 temporadas casi completas."""
    import numpy as np
    t = [x for x in _serie() if x["partidos"] >= 900][-ANOS_TENDENCIA:]
    return float(np.polyfit([x["temporada"] for x in t], [x["media"] for x in t], 1)[0]) if len(t) >= 3 else 0.0


@app.post("/predict/fecha")
def predecir_fecha(p: PartidoFecha):
    from estado_equipos import forma_hasta, temporada_de
    from modelo_puntos import media_liga_actual
    import pandas as pd
    est = _estado()
    nombres = {x["id"]: x["nombre"] for x in est["equipos"]}
    if p.local_id == p.visitante_id:
        raise HTTPException(422, "El local y el visitante tienen que ser equipos distintos")
    for i in (p.local_id, p.visitante_id):
        if i not in nombres:
            raise HTTPException(404, f"Equipo {i} no encontrado; consulta /equipos")
    fecha = pd.Timestamp(p.fecha)
    corte = pd.Timestamp(est["fecha_corte"])
    futuro = fecha > corte
    real, aviso, tendencia, ajuste = None, None, None, 0.0
    if futuro:
        por_id = {x["id"]: x for x in est["equipos"]}
        lo, vi = dict(por_id[p.local_id]), dict(por_id[p.visitante_id])
        for f in (lo, vi):   # descanso = días desde su último partido hasta la fecha elegida (tope 7)
            f["dias_descanso"] = int(min(7, (fecha - pd.Timestamp(f["ultimo_partido"])).days))
        ref = est["media_liga_reciente"]
        datos_hasta = est["fecha_corte"]
        temporada = max(est["temporada"], temporada_de(fecha))
        aviso = (f"Estimación: ese partido aún no se ha jugado. Se usa la forma más reciente conocida (datos hasta {datos_hasta}); "
                 "cuanto más lejos esté la fecha, menos fiable es.")
        if p.con_tendencia:
            # La liga cada vez anota más: sumamos a la media de liga (y a las dos sumas de puntos de las entradas, que son
            # absolutas) lo que subiría según la tendencia reciente. Así el modelo ve la misma situación "relativa" que ya conoce.
            anos = min(max((fecha - corte).days / 365.25, 0), MAX_ANOS_EXTRAPOLAR)
            sube = _pendiente()
            ajuste = round(sube * anos, 1)
            ref = round(ref + ajuste, 2)
            tendencia = {"puntos_por_ano": round(sube, 2), "anos_aplicados": round(anos, 1), "ajuste": ajuste}
            aviso += (f" Se aplica la tendencia de la liga (+{tendencia['puntos_por_ano']} puntos por año en las últimas {ANOS_TENDENCIA} temporadas), "
                      f"que suma {ajuste} puntos al nivel actual" + (f"; más allá de {MAX_ANOS_EXTRAPOLAR} años la tendencia se detiene." if (fecha - corte).days / 365.25 > MAX_ANOS_EXTRAPOLAR else "."))
    else:
        h = _historial()
        lo, vi = forma_hasta(h, p.local_id, fecha), forma_hasta(h, p.visitante_id, fecha)
        for f, i in ((lo, p.local_id), (vi, p.visitante_id)):
            if f is None:
                raise HTTPException(422, f"{nombres[i]} no tiene 10 partidos de temporada regular antes del {p.fecha}. Prueba con una fecha posterior.")
        previos = h[(h.es_local == 1) & (h.game_date < fecha)].sort_values(["game_date", "game_id"])
        ref = round(media_liga_actual((previos.pts_favor + previos.pts_contra).values), 2)
        datos_hasta = str(max(pd.Timestamp(lo["ultimo_partido"]), pd.Timestamp(vi["ultimo_partido"])).date())
        temporada = temporada_de(fecha)
        r = h[(h.game_date == fecha) & (h.equipo == p.local_id) & (h.rival == p.visitante_id) & (h.es_local == 1)]
        if len(r):
            r = r.iloc[0]
            real = {"puntos_local": int(r.pts_favor), "puntos_visitante": int(r.pts_contra), "puntos_totales": int(r.pts_favor + r.pts_contra),
                    "ganador": "local" if r.pts_favor > r.pts_contra else "visitante"}
        else:
            aviso = ("Ese día esos dos equipos no jugaron entre sí (o no era temporada regular), así que no hay resultado real con que comparar. "
                     "La predicción usa su forma justo antes de esa fecha.")
    ent = _entradas(lo, vi, lo["dias_descanso"], vi["dias_descanso"])
    ganador = _pred_ganador(_df(ent))                 # el ganador solo usa diferencias: no depende del nivel de puntos
    if ajuste:
        for c in ("suma_pts_favor_ult10", "suma_pts_contra_ult10"):
            ent[c] += ajuste
    puntos = _pred_puntos(_df(ent), ref)
    if real:
        real["acierta_ganador"] = ganador["ganador"] == real["ganador"]
        real["error_puntos"] = round(puntos["puntos_totales"] - real["puntos_totales"], 1)
    return {"local": nombres[p.local_id], "visitante": nombres[p.visitante_id], "fecha": str(p.fecha), "modo": "futuro" if futuro else "pasado",
            "datos_hasta": datos_hasta, "temporada": temporada, "aviso": aviso, "real": real, "tendencia": tendencia,
            "dias_descanso": {"local": lo["dias_descanso"], "visitante": vi["dias_descanso"]},
            "entradas": _limpio(ent), "ganador": ganador, "puntos": puntos}


@app.get("/serie-puntos")
def serie_puntos():
    """Puntos totales medios por temporada (los de docs/drift/drift_por_temporada.csv), para dibujar el drift."""
    filas = _serie()
    est = _cache.get("estado") or (json.loads(_ruta("estado_equipos.json").read_text(encoding="utf-8")) if _ruta("estado_equipos.json").exists() else {})
    return {"temporadas": filas, "entrenamiento_hasta": 2022, "lote2_desde": 2024, "media_liga_reciente": est.get("media_liga_reciente")}


app.mount("/static", StaticFiles(directory=RAIZ / "api" / "static"), name="static")


@app.get("/", include_in_schema=False)
def inicio():
    return FileResponse(RAIZ / "api" / "static" / "index.html")
