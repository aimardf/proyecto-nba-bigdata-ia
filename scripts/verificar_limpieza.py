#!/usr/bin/env python3
"""
verificar_limpieza.py · HoopMind (Big Data + IA)

Comprueba que la limpieza de Hive recupera los datos originales. Compara tres cosas:

  1. raw       Games.csv y TeamStatistics.csv originales (zona /nba/raw)
  2. log       ensuciado_log.csv (que celdas se ensuciaron y con que valor original)
  3. curated   exportacion a CSV de nba_curated.partidos y nba_curated.estadisticas_equipo

Saca dos tablas:
  A. Por problema: cuantas celdas ensuciadas vuelven a su valor original (y, si no,
     el error medio). Los nulos que no se pueden deducir se imputan con una media
     movil, asi que ahi no se espera el 100 %: se informa el error absoluto medio.
  B. Danos colaterales: celdas que NO se ensuciaron y que la limpieza ha cambiado.
     Lo ideal es 0; si sale algo, hay que mirarlo (puede ser un problema del dato
     original que la limpieza ha corregido a proposito, o un fallo de la limpieza).

Uso:
  python3 scripts/verificar_limpieza.py --raw data/raw_2026-11-17 \
      --log data/dirty/ensuciado_log.csv --curated data/curated_export \
      --informe logs/verificacion_limpieza.csv
"""
import argparse
import pathlib
import re
import sys

import numpy as np
import pandas as pd

# raw -> curated. Las columnas que no estan aqui no se comparan.
MAPA = {
    "games": {
        "fichero_raw": "Games.csv", "fichero_cur": "partidos.csv",
        "claves": {"gameId": "game_id"},
        "columnas": {
            "gameDateTimeEst": "game_datetime", "gameType": "game_type",
            "hometeamCity": "home_team_city", "hometeamName": "home_team_name", "hometeamId": "home_team_id",
            "awayteamCity": "away_team_city", "awayteamName": "away_team_name", "awayteamId": "away_team_id",
            "homeScore": "home_score", "awayScore": "away_score", "winner": "winner",
            "attendance": "attendance", "arenaId": "arena_id", "arenaName": "arena_name",
        },
    },
    "team_statistics": {
        "fichero_raw": "TeamStatistics.csv", "fichero_cur": "estadisticas_equipo.csv",
        "claves": {"gameId": "game_id", "teamId": "team_id"},
        "columnas": {},  # se rellena abajo con camelCase -> snake_case
    },
}
TS_COMPARAR = [
    "gameDateTimeEst", "teamCity", "teamName", "opponentTeamId", "home", "win", "teamScore",
    "opponentScore", "assists", "blocks", "steals", "fieldGoalsAttempted", "fieldGoalsMade",
    "fieldGoalsPercentage", "threePointersAttempted", "threePointersMade",
    "threePointersPercentage", "freeThrowsAttempted", "freeThrowsMade", "freeThrowsPercentage",
    "reboundsDefensive", "reboundsOffensive", "reboundsTotal", "foulsPersonal", "turnovers",
    "plusMinusPoints", "q1Points", "q2Points", "q3Points", "q4Points", "ot1Points", "ot2Points",
    "otAllPoints", "benchPoints", "pointsInThePaint",
]
RENOMBRE_TS = {"gameDateTimeEst": "game_datetime"}
VACIO_ES_CERO = {"ot1Points", "ot2Points", "otAllPoints"}   # regla de la limpieza: sin prorroga = 0
TOL_PCT = 0.0011    # los % vienen redondeados a 3 decimales


def snake(nombre):
    return re.sub(r"(?<!^)(?=[A-Z])", "_", nombre).lower()


for c in TS_COMPARAR:
    MAPA["team_statistics"]["columnas"][c] = RENOMBRE_TS.get(c, snake(c))


def leer(ruta):
    if not ruta.exists():
        sys.exit(f"ERROR: no encuentro {ruta}")
    df = pd.read_csv(ruta, dtype=str, keep_default_na=False, na_filter=False)
    # beeline puede poner el nombre de la tabla delante (partidos.game_id)
    df.columns = [c.split(".")[-1] for c in df.columns]
    return df


def a_numero(s):
    return pd.to_numeric(s.replace({"NULL": "", "null": ""}), errors="coerce")


def norm_fecha(s):
    # Hive escribe '2026-11-15 21:00:00.0'; el original '2026-11-15 21:00:00'
    return s.str.replace(r"\.0+$", "", regex=True).str.strip()


def iguales(orig, cur, col_raw):
    """Serie booleana: el valor curado es igual al original (con tolerancia en numeros)."""
    o, c = orig.copy(), cur.copy()
    if col_raw in VACIO_ES_CERO:
        o = o.mask(o.str.strip() == "", "0")
    if col_raw == "gameDateTimeEst":
        return norm_fecha(o) == norm_fecha(c)
    on, cn = a_numero(o), a_numero(c)
    numerico = on.notna() | cn.notna()
    tol = TOL_PCT if col_raw.endswith("Percentage") else 1e-6
    ig_num = (on - cn).abs() <= tol
    ambos_nulos = on.isna() & cn.isna() & (o.str.strip().isin(["", "NULL"])) & (c.str.strip().isin(["", "NULL"]))
    ig_txt = o.str.strip() == c.replace({"NULL": ""}).str.strip()
    return np.where(numerico, ig_num | ambos_nulos, ig_txt)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--curated", required=True)
    ap.add_argument("--informe", default=None, help="CSV donde guardar la tabla A")
    args = ap.parse_args()

    log = leer(pathlib.Path(args.log))
    filas_a, filas_b = [], []

    for tabla, m in MAPA.items():
        raw = leer(pathlib.Path(args.raw) / m["fichero_raw"])
        cur = leer(pathlib.Path(args.curated) / m["fichero_cur"])
        k_raw, k_cur = list(m["claves"]), list(m["claves"].values())

        # --- claves y duplicados ---
        for kr, kc in m["claves"].items():
            raw[kr] = raw[kr].str.strip()
            cur[kc] = a_numero(cur[kc]).astype("Int64").astype(str)
        dup_cur = int(cur.duplicated(k_cur).sum())
        raw_u = raw.drop_duplicates(k_raw)
        # prefijo raw: para que no choquen columnas que se llaman igual en los dos lados (assists, winner...)
        raw_u = raw_u.add_prefix("raw:")
        k_raw = ["raw:" + k for k in k_raw]
        unido = raw_u.merge(cur, left_on=k_raw, right_on=k_cur, how="left", indicator=True)
        faltan = int((unido["_merge"] == "left_only").sum())
        sobran = len(cur) - (len(unido) - faltan)
        print(f"\n=== {tabla} ===")
        print(f"Filas raw (sin repetir clave): {len(raw_u)} | filas curated: {len(cur)} | "
              f"duplicados en curated: {dup_cur} | de raw que faltan en curated: {faltan} | "
              f"de curated que no estan en raw: {sobran}")
        k_log = len(k_raw)
        filas_a.append({"tabla": tabla, "problema": "2_duplicados",
                        "celdas": int((log.tabla.eq(tabla) & log.problema.eq("2_duplicados")).sum()),
                        "recuperadas": None, "pct_recuperado": 100.0 if dup_cur == 0 else 0.0,
                        "error_medio_no_recuperadas": None,
                        "nota": f"{dup_cur} claves repetidas en curated"})
        unido = unido[unido["_merge"] == "both"]

        # --- celdas ensuciadas ---
        lt = log[(log.tabla == tabla) & (log.columna.isin(m["columnas"]))].copy()
        lt["clave"] = lt["gameId"].str.strip() + ("|" + lt["teamId"].str.strip() if k_log == 2 else "")
        unido["clave"] = unido[k_raw[0]] + ("|" + unido[k_raw[1]] if k_log == 2 else "")
        ensuciadas = set(zip(lt["clave"], lt["columna"]))
        idx = unido.set_index("clave")

        for (problema), g in lt.groupby("problema"):
            ok, err, n = 0, [], 0
            for col_raw, gg in g.groupby("columna"):
                col_cur = m["columnas"][col_raw]
                presentes = gg[gg["clave"].isin(idx.index)]
                n += len(gg)
                if presentes.empty:
                    continue
                o = presentes["valor_original"].reset_index(drop=True)
                c = idx.loc[presentes["clave"], col_cur].reset_index(drop=True).astype(str)
                ig = iguales(o, c, col_raw)
                ok += int(ig.sum())
                d = (a_numero(o) - a_numero(c)).abs()[~ig].dropna()
                err.extend(d.tolist())
            filas_a.append({"tabla": tabla, "problema": problema, "celdas": n, "recuperadas": ok,
                            "pct_recuperado": round(100 * ok / n, 2) if n else None,
                            "error_medio_no_recuperadas": round(float(np.mean(err)), 4) if err else None,
                            "nota": ""})

        # --- danos colaterales: celdas no ensuciadas que cambian ---
        for col_raw, col_cur in m["columnas"].items():
            if col_cur not in unido.columns:
                print(f"AVISO: curated no tiene la columna {col_cur}")
                continue
            limpias = ~pd.Series([(k, col_raw) in ensuciadas for k in unido["clave"]], index=unido.index)
            u = unido[limpias]
            ig = iguales(u["raw:" + col_raw].astype(str), u[col_cur].astype(str), col_raw)
            distintos = int((~ig).sum())
            if distintos:
                ej = u.loc[~ig, [k_raw[0], "raw:" + col_raw, col_cur]].head(3).values.tolist()
                filas_b.append({"tabla": tabla, "columna": col_raw, "celdas_cambiadas": distintos,
                                "de": len(u), "ejemplos (clave, raw, curated)": ej})

    a = pd.DataFrame(filas_a)
    print("\nA. RECUPERACION DE LO ENSUCIADO")
    print(a.to_string(index=False))
    print("\nB. DANOS COLATERALES (celdas no ensuciadas que la limpieza ha cambiado)")
    if filas_b:
        print(pd.DataFrame(filas_b).to_string(index=False))
    else:
        print("Ninguno: todas las celdas no ensuciadas siguen igual que en raw.")
    if args.informe:
        pathlib.Path(args.informe).parent.mkdir(parents=True, exist_ok=True)
        a.to_csv(args.informe, index=False)
        if filas_b:
            pd.DataFrame(filas_b).to_csv(str(args.informe).replace(".csv", "_colaterales.csv"), index=False)
        print(f"\nInforme guardado en {args.informe}")


if __name__ == "__main__":
    main()
