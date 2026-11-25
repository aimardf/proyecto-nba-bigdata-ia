"""Pruebas de la API con modelos de juguete entrenados al vuelo (no hacen falta datos reales).
Ejecutar desde la raíz del repo:  python -m pytest api/test_api.py -q
"""
import sys
import pathlib

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
from modelo_ganador import guardar_modelo_ganador  # noqa: E402
from modelo_puntos import ModeloPuntosRelativo, guardar_modelo  # noqa: E402


def partidos_de_juguete(n_equipos=8, temporadas=(2022,2023,2024), seed=1):
    rng=np.random.default_rng(seed); filas=[]; gid=0
    for t in temporadas:
        d0=pd.Timestamp(f"{t}-10-25")
        for dia in range(0,150):
            eqs=rng.permutation(n_equipos)
            for i in range(0,n_equipos,2):
                h,a=int(eqs[i])+1,int(eqs[i+1])+1; gid+=1
                filas.append(dict(game_id=gid,game_date=d0+pd.Timedelta(days=dia),temporada=t,game_type="Regular Season",jugado=1,
                  home_team_id=h,away_team_id=a,home_score=int(rng.normal(112,10)),away_score=int(rng.normal(108,10)),
                  home_team_city=f"Ciudad{h}",home_team_name=f"Eq{h}",away_team_city=f"Ciudad{a}",away_team_name=f"Eq{a}"))
    g=pd.DataFrame(filas); e=[]
    for r in g.itertuples():
        for tid in (r.home_team_id,r.away_team_id):
            e.append(dict(game_id=r.game_id,team_id=tid,field_goals_percentage=rng.uniform(.4,.5),three_pointers_percentage=rng.uniform(.3,.4),
              free_throws_percentage=rng.uniform(.7,.8),rebounds_total=rng.integers(35,50),assists=rng.integers(20,30),turnovers=rng.integers(10,18),
              steals=rng.integers(5,10),blocks=rng.integers(3,7)))
    return g,pd.DataFrame(e)


@pytest.fixture()
def cliente(tmp_path, monkeypatch):
    monkeypatch.setenv("HOOPMIND_MODELOS", str(tmp_path))
    from api import main
    main._cache.clear()
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(0, 1, (400, 15)), columns=main.COLUMNAS)
    g = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression()).fit(X, (X.iloc[:, 0] > 0).astype(int))
    guardar_modelo_ganador(g, main.COLUMNAS, tmp_path / "modelo_ganador.joblib")
    ref = np.full(400, 225.0)
    guardar_modelo(ModeloPuntosRelativo("ridge").fit(X, 225 + X.iloc[:, 2] * 3, ref), tmp_path / "modelo_puntos.joblib")
    import json
    from estado_equipos import calcular, historial
    from construir_modelado import largo_equipos
    g, e = partidos_de_juguete()
    (tmp_path / "estado_equipos.json").write_text(json.dumps(calcular(g, e)), encoding="utf-8")
    historial(largo_equipos(g, e)[1]).to_csv(tmp_path / "historial_equipos.csv.gz", index=False, compression="gzip")
    return TestClient(main.app), main


def test_health_y_contrato(cliente):
    c, main = cliente
    assert c.get("/health").json() == {"estado": "ok", "modelo_ganador": True, "modelo_puntos": True}
    assert c.get("/contrato").json()["entradas"] == main.COLUMNAS and len(main.COLUMNAS) == 15


def test_ganador(cliente):
    c, main = cliente
    r = c.post("/predict/ganador", json={k: 1.0 for k in main.COLUMNAS})
    assert r.status_code == 200 and 0 <= r.json()["prob_local"] <= 1 and r.json()["ganador"] in ("local", "visitante")


def test_puntos_respeta_tope(cliente):
    c, main = cliente
    r = c.post("/predict/puntos", json={**{k: 0.0 for k in main.COLUMNAS}, "media_liga_reciente": 229.3})
    assert r.status_code == 200 and 127 <= r.json()["puntos_totales"] <= 319


def test_faltan_entradas(cliente):
    c, main = cliente
    assert c.post("/predict/ganador", json={"dif_pts_favor_ult10": 1}).status_code == 422


def test_sin_modelos_da_503(tmp_path, monkeypatch):
    monkeypatch.setenv("HOOPMIND_MODELOS", str(tmp_path))
    from api import main
    main._cache.clear()
    c = TestClient(main.app)
    assert c.post("/predict/ganador", json={k: 0.0 for k in main.COLUMNAS}).status_code == 503


def test_pagina_inicio(cliente):
    assert "HoopMind" in cliente[0].get("/").text


def test_equipos_y_partido(cliente):
    c, main = cliente
    eq = c.get("/equipos").json()
    assert len(eq["equipos"]) == 8 and eq["datos_hasta"]
    assert "pts_favor_ult10" in eq["equipos"][0]["forma"] and "win_pct_temporada" in eq["equipos"][0]["forma"]
    r = c.post("/predict/partido", json={"local_id": 1, "visitante_id": 2})
    assert r.status_code == 200
    j = r.json()
    assert set(j["entradas"]) == set(main.COLUMNAS)
    assert j["ganador"]["ganador"] in ("local", "visitante") and 127 <= j["puntos"]["puntos_totales"] <= 319


def test_partido_errores(cliente):
    c, _ = cliente
    assert c.post("/predict/partido", json={"local_id": 1, "visitante_id": 1}).status_code == 422
    assert c.post("/predict/partido", json={"local_id": 1, "visitante_id": 999}).status_code == 404


def test_estado_coincide_con_la_tabla_de_modelado():
    """Las entradas que calcula la API para un partido nuevo son las mismas que daría construir_modelado.py."""
    import construir_modelado as cm
    from estado_equipos import calcular
    g, e = partidos_de_juguete()
    est = {x["id"]: x for x in calcular(g, e)["equipos"]}
    nuevo = g.iloc[-1:].copy()
    nuevo["game_id"], nuevo["game_date"] = 99999, g.game_date.max() + pd.Timedelta(days=2)
    nuevo["home_team_id"], nuevo["away_team_id"] = 1, 2
    ne = pd.DataFrame([dict(game_id=99999, team_id=t, field_goals_percentage=.4, three_pointers_percentage=.3,
                            free_throws_percentage=.7, rebounds_total=40, assists=20, turnovers=12, steals=6, blocks=4) for t in (1, 2)])
    fila = cm.construir(pd.concat([g, nuevo]), pd.concat([e, ne])).query("game_id == 99999").iloc[0]
    for m in ("pts_favor", "pts_contra", "win_pct", "rebotes", "tapones"):
        assert abs((est[1][f"{m}_ult10"] - est[2][f"{m}_ult10"]) - fila[f"dif_{m}_ult10"]) < 1e-3
    assert abs((est[1]["win_pct_temporada"] - est[2]["win_pct_temporada"]) - fila["dif_win_pct_temporada"]) < 1e-9


def test_sin_estado_da_503(tmp_path, monkeypatch):
    monkeypatch.setenv("HOOPMIND_MODELOS", str(tmp_path))
    from api import main
    main._cache.clear()
    assert TestClient(main.app).get("/equipos").status_code == 503


def test_fecha_pasada_con_resultado_real(cliente):
    c, main = cliente
    g, _ = partidos_de_juguete()
    f = g[g.game_date >= "2023-02-01"].iloc[0]          # un partido real del histórico de juguete
    r = c.post("/predict/fecha", json={"local_id": int(f.home_team_id), "visitante_id": int(f.away_team_id), "fecha": str(f.game_date.date())})
    assert r.status_code == 200
    j = r.json()
    assert j["modo"] == "pasado" and j["real"]["puntos_totales"] == f.home_score + f.away_score
    assert j["real"]["acierta_ganador"] in (True, False) and 127 <= j["puntos"]["puntos_totales"] <= 319
    assert j["datos_hasta"] < j["fecha"]                 # nunca usa el propio día (sin fuga)


def test_fecha_futura_es_estimacion(cliente):
    c, _ = cliente
    r = c.post("/predict/fecha", json={"local_id": 1, "visitante_id": 2, "fecha": "2031-01-15"})
    j = r.json()
    assert r.status_code == 200 and j["modo"] == "futuro" and j["real"] is None and "Estimación" in j["aviso"]
    assert j["dias_descanso"] == {"local": 7, "visitante": 7}


def test_fecha_demasiado_antigua_da_422(cliente):
    c, _ = cliente
    assert c.post("/predict/fecha", json={"local_id": 1, "visitante_id": 2, "fecha": "2000-01-01"}).status_code == 422
    assert c.post("/predict/fecha", json={"local_id": 1, "visitante_id": 1, "fecha": "2031-01-15"}).status_code == 422


def test_serie_puntos(cliente):
    j = cliente[0].get("/serie-puntos").json()
    assert j["temporadas"][0]["temporada"] == 2000 and j["lote2_desde"] == 2024


def test_forma_hasta_coincide_con_la_tabla_de_modelado():
    """La forma en una fecha pasada = la fila de construir_modelado.py de ese partido."""
    import construir_modelado as cm
    from estado_equipos import forma_hasta, historial
    g, e = partidos_de_juguete()
    h = historial(cm.largo_equipos(g, e)[1])
    ds = cm.construir(g, e)
    fila = ds[ds.game_date >= "2023-03-01"].iloc[0]
    lo, vi = forma_hasta(h, int(fila.home_team_id), fila.game_date), forma_hasta(h, int(fila.away_team_id), fila.game_date)
    for m in ("pts_favor", "pts_contra", "win_pct", "rebotes", "tapones"):
        assert abs((lo[f"{m}_ult10"] - vi[f"{m}_ult10"]) - fila[f"dif_{m}_ult10"]) < 1e-2
    assert abs((lo["win_pct_temporada"] - vi["win_pct_temporada"]) - fila["dif_win_pct_temporada"]) < 1e-3
    assert lo["dias_descanso"] - vi["dias_descanso"] == fila["dif_dias_descanso"]


def test_futuro_con_y_sin_tendencia(cliente):
    c, _ = cliente
    base = {"local_id": 1, "visitante_id": 2, "fecha": "2036-01-15"}
    con = c.post("/predict/fecha", json=base).json()
    sin = c.post("/predict/fecha", json={**base, "con_tendencia": False}).json()
    assert sin["tendencia"] is None and con["tendencia"]["puntos_por_ano"] > 0
    assert con["puntos"]["media_liga_usada"] > sin["puntos"]["media_liga_usada"]      # la liga "sube"
    assert con["puntos"]["puntos_totales"] >= sin["puntos"]["puntos_totales"]
    lejos = c.post("/predict/fecha", json={**base, "fecha": "2061-01-15"}).json()
    assert lejos["tendencia"]["anos_aplicados"] == 10.0                                  # tope de extrapolación
