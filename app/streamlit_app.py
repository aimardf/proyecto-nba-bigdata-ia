"""HoopMind · Panel Streamlit para explorar los datos y los modelos.

Solo LEE ficheros del repo (no cambia nada) y, si la API está encendida, le pregunta una predicción.

Arrancar (desde la raíz del repo):
  pip install streamlit plotly
  streamlit run app/streamlit_app.py
"""
import os
import pathlib

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

RAIZ = pathlib.Path(__file__).resolve().parent.parent
DOCS = RAIZ / "docs"
PARQUET = RAIZ / "data" / "modelado_partidos.parquet"
API = os.environ.get("HOOPMIND_API", "http://127.0.0.1:8000")

AZUL, NARANJA, AZUL_CL, CREMA = "#174D75", "#D9702B", "#DCE7EF", "#F8F5E4"
PALETA = [AZUL, NARANJA]

st.set_page_config(page_title="HoopMind · Panel de datos", page_icon="🏀", layout="wide")
st.markdown(f"""<style>
h1, h2, h3 {{ color:{AZUL}; }}
[data-testid="stMetricValue"] {{ color:{AZUL}; }}
[data-testid="stSidebar"] {{ background:{CREMA}; }}
.titulo span {{ color:{NARANJA}; }}
</style>""", unsafe_allow_html=True)


@st.cache_data
def leer_csv(ruta):
    return pd.read_csv(ruta)


@st.cache_data
def leer_partidos():
    if not PARQUET.exists():
        return None
    return pd.read_parquet(PARQUET)


def figura(fig):
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor="white", legend_title_text="")
    return fig


def mostrar(fig):
    st.plotly_chart(figura(fig), width="stretch")


partidos = leer_partidos()
ENTRADAS = [c for c in (partidos.columns if partidos is not None else [])
            if c.startswith(("dif_", "suma_"))]

st.sidebar.markdown("## HoopMind")
st.sidebar.caption("Predicción de partidos NBA con datos de equipo (sin jugadores). Big Data + IA.")
seccion = st.sidebar.radio("Sección", ["1. Datos", "2. Puntos por temporada", "3. Entradas del modelo",
                                       "4. Correlaciones", "5. Resultados de modelos", "6. Predecir un partido"])
if partidos is None:
    st.sidebar.warning("No encuentro data/modelado_partidos.parquet. Uso los resúmenes de docs/ donde se puede.")

st.markdown("<h1 class='titulo'>Hoop<span>Mind</span> · Panel de datos</h1>", unsafe_allow_html=True)

# ---------------------------------------------------------------- 1. Datos
if seccion.startswith("1"):
    st.header("Datos del proyecto")
    if partidos is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Partidos", f"{len(partidos):,}".replace(",", "."))
        c2.metric("Temporadas", f"{int(partidos.temporada.min())}–{int(partidos.temporada.max())}")
        c3.metric("% gana el local", f"{partidos.home_win.mean():.1%}".replace(".", ","))
        c4.metric("Puntos totales (media)", f"{partidos.puntos_totales.mean():.1f}".replace(".", ","))
        st.caption("Cada fila es un partido. Las entradas (dif_… / suma_…) se calculan con lo que había ANTES del partido; "
                   "las dos salidas son home_win (ganador) y puntos_totales.")
        temp = st.multiselect("Filtrar temporadas", sorted(partidos.temporada.unique()), default=[])
        vista = partidos[partidos.temporada.isin(temp)] if temp else partidos
        st.dataframe(vista.head(500), width="stretch")
        st.caption(f"Se muestran las primeras 500 filas de {len(vista):,}.".replace(",", "."))
        with st.expander("Resumen estadístico"):
            st.dataframe(vista.select_dtypes("number").describe().T, width="stretch")
    else:
        st.info("Sin el parquet no puedo mostrar los partidos. Genéralo con scripts/construir_modelado.py.")
        st.dataframe(leer_csv(DOCS / "eda" / "resumen_por_temporada.csv"), width="stretch")

# ---------------------------------------------------------------- 2. Puntos por temporada
elif seccion.startswith("2"):
    st.header("Evolución de los puntos totales por temporada")
    if partidos is not None:
        t = partidos.groupby("temporada").agg(media=("puntos_totales", "mean"), partidos=("game_id", "count")).reset_index()
    else:
        t = leer_csv(DOCS / "eda" / "resumen_por_temporada.csv").rename(columns={"puntos": "media"})
    fig = px.line(t, x="temporada", y="media", markers=True, color_discrete_sequence=[AZUL],
                  labels={"media": "Puntos totales medios", "temporada": "Temporada"})
    fig.update_traces(hovertemplate="%{x}: %{y:.1f} puntos")
    mostrar(fig)
    st.info(f"La media pasa de unos {t.media.iloc[0]:.0f} puntos ({int(t.temporada.iloc[0])}) a {t.media.iloc[-1]:.0f} "
            f"({int(t.temporada.iloc[-1])}). Es el *drift* del objetivo 2: el modelo debe predecir la desviación "
            "respecto a la media reciente de la liga, no el número absoluto.")
    d = leer_csv(DOCS / "drift" / "drift_por_temporada.csv")
    with st.expander("Tabla de drift por temporada (PSI)"):
        st.dataframe(d, width="stretch")
    mostrar(px.bar(d, x="temporada", y=["PSI_objetivo_vs_train", "PSI_objetivo_relativo"],
                   barmode="group", color_discrete_sequence=PALETA,
                   title="PSI del objetivo: absoluto (azul) vs relativo a la liga (naranja)"))
    st.caption("PSI > 0,25 = cambio grande. El objetivo absoluto se dispara; el relativo se mantiene estable.")

# ---------------------------------------------------------------- 3. Entradas
elif seccion.startswith("3"):
    st.header("Distribución de las entradas")
    if partidos is not None:
        col = st.selectbox("Entrada", ENTRADAS)
        reciente = partidos.temporada >= partidos.temporada.max() - 1
        ver = partidos.assign(grupo=reciente.map({True: "Últimas 2 temporadas", False: "Histórico"}))
        mostrar(px.histogram(ver, x=col, color="grupo", barmode="overlay", opacity=.65, histnorm="probability density",
                             nbins=60, color_discrete_sequence=PALETA))
    else:
        st.info("Sin el parquet no puedo dibujar histogramas.")
    st.subheader("¿Han cambiado las entradas? (drift)")
    st.dataframe(leer_csv(DOCS / "drift" / "drift_entradas_y_objetivo.csv"), width="stretch")
    st.caption("PSI < 0,1 estable · 0,1–0,25 vigilar · > 0,25 reentrenar.")

# ---------------------------------------------------------------- 4. Correlaciones
elif seccion.startswith("4"):
    st.header("Correlaciones con los dos objetivos")
    c = leer_csv(DOCS / "eda" / "correlacion_con_salidas.csv").rename(columns={"Unnamed: 0": "variable"})
    c = c.rename(columns={c.columns[0]: "variable"})
    largo = c.melt("variable", var_name="objetivo", value_name="correlacion")
    largo["objetivo"] = largo.objetivo.map({"home_win": "Ganador (home_win)", "puntos_totales": "Puntos totales"})
    mostrar(px.bar(largo, y="variable", x="correlacion", color="objetivo", barmode="group", orientation="h",
                   color_discrete_sequence=PALETA, height=620))
    st.caption("Las entradas explican algo el ganador (sobre todo el % de victorias reciente) y casi nada los puntos totales: "
               "por eso el modelo de puntos mejora poco sobre la media.")
    if partidos is not None:
        with st.expander("Matriz de correlación entre entradas"):
            mostrar(px.imshow(partidos[ENTRADAS].corr(), color_continuous_scale=[AZUL, "white", NARANJA],
                              zmin=-1, zmax=1, height=700))

# ---------------------------------------------------------------- 5. Modelos
elif seccion.startswith("5"):
    st.header("Resultados de los modelos")
    r = leer_csv(DOCS / "resultados_modelos_base.csv")
    # El CSV puede traer filas de validación cruzada (cv_...) o repetidas: nos quedamos con una fila por modelo.
    r = r[~r.modelo.str.startswith("cv_")].drop_duplicates(["experimento", "modelo"], keep="last")
    st.subheader("Objetivo 1 · ¿Gana el local?")
    o1 = r[r.experimento.str.contains("obj1")][["modelo", "prueba_accuracy", "lote2_accuracy", "prueba_auc", "lote2_auc"]]
    st.dataframe(o1.set_index("modelo"), width="stretch")
    mostrar(px.bar(o1.melt("modelo", ["prueba_accuracy", "lote2_accuracy"], var_name="conjunto", value_name="accuracy"),
                   x="modelo", y="accuracy", color="conjunto", barmode="group", color_discrete_sequence=PALETA,
                   title="Acierto: prueba vs lote 2 (temporadas recientes)"))
    st.subheader("Objetivo 2 · Puntos totales")
    o2 = r[r.experimento.str.contains("obj2")][["modelo", "prueba_mae", "lote2_mae", "prueba_r2", "lote2_r2"]]
    st.dataframe(o2.set_index("modelo"), width="stretch")
    mostrar(px.bar(o2.melt("modelo", ["prueba_mae", "lote2_mae"], var_name="conjunto", value_name="MAE"),
                   x="modelo", y="MAE", color="conjunto", barmode="group", color_discrete_sequence=PALETA,
                   title="Error medio (MAE, puntos): menos es mejor"))
    st.divider()
    st.subheader("Validación temporal (TimeSeriesSplit)")
    st.dataframe(leer_csv(DOCS / "validacion_temporal" / "resumen_cv.csv"), width="stretch")
    m = leer_csv(DOCS / "validacion_temporal" / "metricas_por_corte.csv")
    ob = st.radio("Objetivo", ["Ganador (accuracy)", "Puntos (MAE)"], horizontal=True)
    y = "accuracy" if ob.startswith("Ganador") else "mae"
    sub = m[m.experimento.str.contains("obj1" if y == "accuracy" else "obj2")]
    fig = px.line(sub, x="corte", y=y, color="modelo", markers=True, color_discrete_sequence=[AZUL, NARANJA, "#6B1F2B", "#7FA6C4", "#F0B27A"])
    fig.update_xaxes(dtick=1)
    mostrar(fig)
    img = DOCS / "validacion_temporal" / "relativo_vs_absoluto_por_corte.png"
    if img.exists():
        st.image(str(img), caption="Predecir lo relativo a la liga vs lo absoluto, por corte temporal")

# ---------------------------------------------------------------- 6. Predicción
else:
    st.header("Predecir un partido")
    st.caption(f"Esta sección le pregunta a la API FastAPI ({API}). Enciéndela con: uvicorn api.main:app")
    try:
        eq = requests.get(f"{API}/equipos", timeout=3).json()
    except Exception:
        st.warning("La API no está encendida (o faltan los modelos). Arráncala en otra terminal y pulsa R para recargar.")
        st.code("uvicorn api.main:app --reload", language="bash")
        st.stop()
    nombres = {e["nombre"]: e["id"] for e in eq["equipos"]}
    st.caption(f"Datos de los equipos hasta {eq['datos_hasta']} (temporada {eq['temporada']}).")
    a, b = st.columns(2)
    local = a.selectbox("Local", sorted(nombres))
    visita = b.selectbox("Visitante", sorted(nombres), index=1)
    d1, d2 = st.columns(2)
    dl = d1.slider("Días de descanso del local", 0, 7, 2)
    dv = d2.slider("Días de descanso del visitante", 0, 7, 2)
    if st.button("Predecir", type="primary"):
        resp = requests.post(f"{API}/predict/partido", timeout=10,
                             json={"local_id": nombres[local], "visitante_id": nombres[visita],
                                   "dias_descanso_local": dl, "dias_descanso_visitante": dv})
        if resp.status_code != 200:
            st.error(resp.json().get("detail", resp.text))
        else:
            res = resp.json()
            st.subheader(f"{res['local']} vs {res['visitante']}")
            st.json({"ganador": res["ganador"], "puntos": res["puntos"]})
            with st.expander("Las 15 entradas que calculó la API"):
                st.dataframe(pd.Series(res["entradas"], name="valor"), width="stretch")
