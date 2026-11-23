#!/usr/bin/env python3
"""
ensuciar.py · HoopMind (Big Data + IA)

Ensucia de forma REPRODUCIBLE los CSV de partido y equipo (Games.csv y
TeamStatistics.csv) para simular los problemas de un dato real. Despues, la
limpieza con Hive (hive/02_limpieza_curated.hql) tiene que recuperar los valores
originales, y scripts/verificar_limpieza.py mide cuanto lo consigue.

Plan de ensuciado (informe v3.2, seccion 6):

  PROBLEMA                      TABLA          CUANTO            COMO SE RESUELVE EN HIVE
  1 Nulos en estadisticas       TeamStatistics 3 % de celdas     Reglas (anotados/intentados/%) y media movil
  2 Filas duplicadas            las dos        1 % de filas      ROW_NUMBER() por clave
  3 Nombres de equipo           las dos        2 % de filas      Nombre mas frecuente por (teamId, temporada)
  4 Fechas en otro formato      las dos        5 % de filas      Parseo con varios formatos (RLIKE)
  5 Valores imposibles          TeamStatistics 0,5 % de filas    Reglas de rango y recalculo
  6 Tipos erroneos («112 pts»)  las dos        10 % de filas     regexp_replace y coma decimal
  7 Local/visitante cambiados   Games          0,3 % de partidos Contraste con la marca home de TeamStatistics
  8 Marcador incoherente        Games          0,5 % de partidos Suma de los cuartos de TeamStatistics

Ademas, por minimizacion de datos (RGPD art. 5.1.c), la zona dirty ya NO lleva
la columna officials (nombres de arbitros) de Games ni coachId de TeamStatistics.

Salidas (en --salida):
  Games.csv, TeamStatistics.csv   datos ensuciados (todo texto)
  ensuciado_log.csv               una fila por cambio: tabla, problema, gameId,
                                  teamId, columna, valor_original, valor_nuevo
  ensuciado_resumen.json          semilla, tasas y cuantos cambios de cada tipo

Uso:
  python3 scripts/ensuciar.py --entrada data/raw_2026-11-17 --salida data/dirty --seed 42
"""
import argparse
import json
import pathlib
import sys
from datetime import datetime

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Configuracion
# --------------------------------------------------------------------------- #
TASAS = {
    "nulos": 0.03,         # de las celdas de estadisticas de equipo
    "duplicados": 0.01,    # de las filas
    "nombres": 0.02,       # de las filas
    "fechas": 0.05,        # de las filas
    "imposibles": 0.005,   # de las filas de TeamStatistics
    "tipos": 0.10,         # de las filas (un campo por fila)
    "intercambio": 0.003,  # de los partidos jugados de Games
    "marcador": 0.005,     # de los partidos jugados de Games
}

# Columnas esperadas, en el orden de los CSV de eoinamoore (capturas del 17/11).
COLS_GAMES = [
    "gameId", "gameDateTimeEst", "hometeamCity", "hometeamName", "hometeamId",
    "awayteamCity", "awayteamName", "awayteamId", "homeScore", "awayScore", "winner",
    "gameType", "gameSubtype", "gameLabel", "gameSubLabel", "seriesGameNumber",
    "attendance", "arenaId", "arenaName", "arenaCity", "arenaState", "officials", "gameDate",
]
COLS_TS = [
    "gameId", "gameDateTimeEst", "teamCity", "teamName", "teamId", "opponentTeamCity",
    "opponentTeamName", "opponentTeamId", "home", "win", "teamScore", "opponentScore",
    "assists", "blocks", "steals", "fieldGoalsAttempted", "fieldGoalsMade",
    "fieldGoalsPercentage", "threePointersAttempted", "threePointersMade",
    "threePointersPercentage", "freeThrowsAttempted", "freeThrowsMade",
    "freeThrowsPercentage", "reboundsDefensive", "reboundsOffensive", "reboundsTotal",
    "foulsPersonal", "turnovers", "plusMinusPoints", "numMinutes", "q1Points", "q2Points",
    "q3Points", "q4Points", "benchPoints", "biggestLead", "biggestScoringRun",
    "leadChanges", "pointsFastBreak", "pointsFromTurnovers", "pointsInThePaint",
    "pointsSecondChance", "timesTied", "timeoutsRemaining", "seasonWins", "seasonLosses",
    "coachId", "gameType", "gameLabel", "gameSubLabel", "seriesGameNumber", "seed",
    "reboundsTeam", "turnoversTeam", "ot1Points", "ot2Points", "otAllPoints", "gameDate",
]
# Minimizacion: columnas con datos de personas que no pasan a dirty ni a curated.
QUITAR = {"games": ["officials"], "team_statistics": ["coachId"]}

# Estadisticas de equipo donde metemos nulos (problema 1)
COLS_NULOS_TS = [
    "assists", "blocks", "steals", "turnovers", "foulsPersonal",
    "fieldGoalsAttempted", "fieldGoalsMade", "fieldGoalsPercentage",
    "threePointersAttempted", "threePointersMade", "threePointersPercentage",
    "freeThrowsAttempted", "freeThrowsMade", "freeThrowsPercentage",
    "reboundsDefensive", "reboundsOffensive", "reboundsTotal",
]
COLS_PCT_TS = ["fieldGoalsPercentage", "threePointersPercentage", "freeThrowsPercentage"]
COLS_CONTEO_TS = ["assists", "blocks", "steals", "turnovers", "foulsPersonal",
                  "reboundsDefensive", "reboundsOffensive", "reboundsTotal"]

# Formatos de fecha "raros" (problema 4). Hive los reconoce por su forma con RLIKE.
FORMATOS_FECHA = ["%d/%m/%Y %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%m-%d-%Y %H:%M:%S"]
FORMATO_ORIGINAL = "%Y-%m-%d %H:%M:%S"


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
class Ensuciador:
    """Guarda el generador aleatorio, las celdas ya tocadas y el log de cambios."""

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.log = []
        self.tocadas = set()  # (tabla, indice, columna)

    def libre(self, tabla, idx, col):
        return (tabla, idx, col) not in self.tocadas

    def cambiar(self, df, tabla, problema, idx, col, nuevo):
        original = df.at[idx, col]
        if nuevo == original:
            return False
        df.at[idx, col] = nuevo
        self.tocadas.add((tabla, idx, col))
        self.log.append({
            "tabla": tabla, "problema": problema,
            "gameId": df.at[idx, "gameId"],
            "teamId": df.at[idx, "teamId"] if "teamId" in df.columns else "",
            "columna": col, "valor_original": original, "valor_nuevo": nuevo,
        })
        return True

    def elegir(self, candidatos, tasa, total=None):
        """Elige round(tasa * total) elementos de candidatos, sin repetir."""
        candidatos = list(candidatos)
        n = int(round(tasa * (total if total is not None else len(candidatos))))
        n = min(n, len(candidatos))
        if n == 0:
            return []
        pos = self.rng.choice(len(candidatos), size=n, replace=False)
        return [candidatos[i] for i in sorted(pos)]


def es_numero(txt):
    try:
        float(txt)
        return True
    except (TypeError, ValueError):
        return False


def leer(ruta, esperadas):
    if not ruta.exists():
        sys.exit(f"ERROR: no encuentro {ruta}")
    # Todo como texto y sin convertir vacios a NaN: asi el log guarda el valor exacto.
    df = pd.read_csv(ruta, dtype=str, keep_default_na=False, na_filter=False)
    faltan = [c for c in esperadas if c not in df.columns]
    if faltan:
        sys.exit(f"ERROR: a {ruta.name} le faltan columnas {faltan}. "
                 "¿Ha cambiado el dataset? Revisa COLS_* en ensuciar.py y el DDL de Hive.")
    sobran = [c for c in df.columns if c not in esperadas]
    if sobran:
        print(f"AVISO: {ruta.name} trae columnas nuevas que se ignoran: {sobran}")
    return df[esperadas].copy()


def parsear_fecha(txt):
    try:
        return datetime.strptime(txt.strip(), FORMATO_ORIGINAL)
    except ValueError:
        return None


def variante_nombre(nombre, rng):
    """Devuelve una forma 'sucia' del nombre: mayusculas, minusculas, espacios o errata."""
    opciones = ["mayus", "minus", "espacios"]
    if len(nombre) >= 4:
        opciones.append("errata")
    tipo = opciones[rng.integers(len(opciones))]
    if tipo == "mayus":
        return nombre.upper()
    if tipo == "minus":
        return nombre.lower()
    if tipo == "espacios":
        return "  " + nombre + " "
    i = int(rng.integers(1, len(nombre) - 2))           # cambia dos letras de sitio
    return nombre[:i] + nombre[i + 1] + nombre[i] + nombre[i + 2:]


# --------------------------------------------------------------------------- #
# Problemas
# --------------------------------------------------------------------------- #
def p1_nulos(df, e, tabla):
    celdas = [(i, c) for c in COLS_NULOS_TS for i in df.index[df[c].to_numpy() != ""]]
    total = len(df) * len(COLS_NULOS_TS)
    for i, c in e.elegir(celdas, TASAS["nulos"], total):
        e.cambiar(df, tabla, "1_nulos", i, c, "")


def p3_nombres(df, e, tabla, cols_nombre):
    filas = e.elegir(list(df.index), TASAS["nombres"])
    for i in filas:
        col = cols_nombre[int(e.rng.integers(len(cols_nombre)))]
        if df.at[i, col].strip() and e.libre(tabla, i, col):
            e.cambiar(df, tabla, "3_nombres", i, col, variante_nombre(df.at[i, col], e.rng))


def p4_fechas(df, e, tabla):
    cands = [i for i in df.index if parsear_fecha(df.at[i, "gameDateTimeEst"]) is not None]
    for i in e.elegir(cands, TASAS["fechas"], len(df)):
        f = parsear_fecha(df.at[i, "gameDateTimeEst"])
        fmt = FORMATOS_FECHA[int(e.rng.integers(len(FORMATOS_FECHA)))]
        e.cambiar(df, tabla, "4_fechas", i, "gameDateTimeEst", f.strftime(fmt))


def p5_imposibles(df, e, tabla):
    filas = e.elegir(list(df.index), TASAS["imposibles"])
    for i in filas:
        if e.rng.random() < 0.5 and es_numero(df.at[i, "teamScore"]) and e.libre(tabla, i, "teamScore"):
            e.cambiar(df, tabla, "5_imposibles", i, "teamScore", str(int(e.rng.integers(350, 451))))
        else:
            col = COLS_PCT_TS[int(e.rng.integers(len(COLS_PCT_TS)))]
            if es_numero(df.at[i, col]) and e.libre(tabla, i, col):
                e.cambiar(df, tabla, "5_imposibles", i, col, f"{e.rng.uniform(1.05, 3.0):.3f}")


def p6_tipos_ts(df, e, tabla):
    """Un campo por fila: «112 pts» en puntos, coma decimal en %, «27.0 » en conteos."""
    filas = e.elegir(list(df.index), TASAS["tipos"])
    for i in filas:
        r = e.rng.random()
        if r < 0.34:
            col = "teamScore"
            nuevo = lambda v: f"{v} pts"
        elif r < 0.67:
            col = COLS_PCT_TS[int(e.rng.integers(len(COLS_PCT_TS)))]
            nuevo = lambda v: v.replace(".", ",")
        else:
            col = COLS_CONTEO_TS[int(e.rng.integers(len(COLS_CONTEO_TS)))]
            nuevo = lambda v: f"{v}.0 "
        v = df.at[i, col]
        # solo valores numericos limpios (sin decimales en conteos y con punto en %)
        if not es_numero(v) or not e.libre(tabla, i, col):
            continue
        if col in COLS_PCT_TS and "." not in v:
            continue
        if col in COLS_CONTEO_TS and "." in v:
            continue
        e.cambiar(df, tabla, "6_tipos", i, col, nuevo(v))


def jugados(df):
    """Indices de partidos con marcador numerico mayor que 0 en los dos equipos."""
    h = pd.to_numeric(df["homeScore"], errors="coerce")
    a = pd.to_numeric(df["awayScore"], errors="coerce")
    return list(df.index[(h > 0) & (a > 0)])


def p7_intercambio(df, e, tabla):
    pares = [("hometeamCity", "awayteamCity"), ("hometeamName", "awayteamName"),
             ("hometeamId", "awayteamId"), ("homeScore", "awayScore")]
    cols = [c for p in pares for c in p]
    cands = [i for i in jugados(df) if all(e.libre(tabla, i, c) for c in cols)]
    for i in e.elegir(cands, TASAS["intercambio"], len(jugados(df))):
        antes = {c: df.at[i, c] for c in cols}
        for h, a in pares:
            e.cambiar(df, tabla, "7_intercambio", i, h, antes[a])
            e.cambiar(df, tabla, "7_intercambio", i, a, antes[h])


def p8_marcador(df, e, tabla):
    j = jugados(df)
    cands = [i for i in j if e.libre(tabla, i, "homeScore") and e.libre(tabla, i, "awayScore")]
    for i in e.elegir(cands, TASAS["marcador"], len(j)):
        col = "homeScore" if e.rng.random() < 0.5 else "awayScore"
        v = int(float(df.at[i, col]))
        delta = int(e.rng.integers(2, 16)) * (1 if e.rng.random() < 0.5 else -1)
        e.cambiar(df, tabla, "8_marcador", i, col, str(max(v + delta, 1)))


def p6_tipos_games(df, e, tabla):
    filas = e.elegir(jugados(df), TASAS["tipos"], len(df))
    for i in filas:
        col = "homeScore" if e.rng.random() < 0.5 else "awayScore"
        v = df.at[i, col]
        if es_numero(v) and "." not in v and e.libre(tabla, i, col):
            e.cambiar(df, tabla, "6_tipos", i, col, f"{v} pts")


def p2_duplicados(df, e, tabla):
    """Copias exactas del 1 % de las filas, metidas en posiciones al azar. Va al final."""
    filas = e.elegir(list(df.index), TASAS["duplicados"])
    for i in filas:
        e.log.append({"tabla": tabla, "problema": "2_duplicados", "gameId": df.at[i, "gameId"],
                      "teamId": df.at[i, "teamId"] if "teamId" in df.columns else "",
                      "columna": "(fila completa)", "valor_original": "", "valor_nuevo": "copia"})
    copias = df.loc[filas]
    juntos = pd.concat([df, copias], ignore_index=True)
    orden = e.rng.permutation(len(juntos))
    return juntos.iloc[orden].reset_index(drop=True)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entrada", required=True, help="carpeta con Games.csv y TeamStatistics.csv originales")
    ap.add_argument("--salida", required=True, help="carpeta donde dejar los CSV ensuciados y el log")
    ap.add_argument("--seed", type=int, default=42, help="semilla (por defecto 42)")
    args = ap.parse_args()

    ent, sal = pathlib.Path(args.entrada), pathlib.Path(args.salida)
    sal.mkdir(parents=True, exist_ok=True)
    e = Ensuciador(args.seed)

    games = leer(ent / "Games.csv", COLS_GAMES)
    ts = leer(ent / "TeamStatistics.csv", COLS_TS)
    filas_orig = {"games": len(games), "team_statistics": len(ts)}

    games = games.drop(columns=QUITAR["games"])
    ts = ts.drop(columns=QUITAR["team_statistics"])

    # Games: el orden importa poco porque cada problema solo usa celdas libres.
    p7_intercambio(games, e, "games")
    p8_marcador(games, e, "games")
    p6_tipos_games(games, e, "games")
    p3_nombres(games, e, "games", ["hometeamName", "awayteamName"])
    p4_fechas(games, e, "games")
    games = p2_duplicados(games, e, "games")

    # TeamStatistics
    p5_imposibles(ts, e, "team_statistics")
    p1_nulos(ts, e, "team_statistics")
    p6_tipos_ts(ts, e, "team_statistics")
    p3_nombres(ts, e, "team_statistics", ["teamName"])
    p4_fechas(ts, e, "team_statistics")
    ts = p2_duplicados(ts, e, "team_statistics")

    games.to_csv(sal / "Games.csv", index=False, lineterminator="\n")
    ts.to_csv(sal / "TeamStatistics.csv", index=False, lineterminator="\n")
    log = pd.DataFrame(e.log, columns=["tabla", "problema", "gameId", "teamId", "columna",
                                       "valor_original", "valor_nuevo"])
    log.to_csv(sal / "ensuciado_log.csv", index=False, lineterminator="\n")

    cuenta = log.groupby(["tabla", "problema"]).size()
    resumen = {
        "semilla": args.seed,
        "fecha_ejecucion": datetime.now().isoformat(timespec="seconds"),
        "tasas": TASAS,
        "columnas_quitadas_minimizacion": QUITAR,
        "filas_originales": filas_orig,
        "filas_ensuciadas": {"games": len(games), "team_statistics": len(ts)},
        "cambios": {f"{t}/{p}": int(n) for (t, p), n in cuenta.items()},
    }
    (sal / "ensuciado_resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False))

    print(f"Semilla {args.seed}. Filas: Games {filas_orig['games']} -> {len(games)}, "
          f"TeamStatistics {filas_orig['team_statistics']} -> {len(ts)}")
    print("Columnas quitadas por minimizacion:", QUITAR)
    print("Cambios por tabla y problema:")
    print(cuenta.to_string())
    print(f"Total: {len(log)} cambios. Log en {sal / 'ensuciado_log.csv'}")


if __name__ == "__main__":
    main()
