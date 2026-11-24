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
