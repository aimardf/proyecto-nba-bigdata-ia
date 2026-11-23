#!/usr/bin/env python3
"""exportar_curated.py · HoopMind

Convierte los Parquet de /nba/curated (copiados de HDFS con `hdfs dfs -get`) a CSV, SIN pasar por Hive.
Es mucho más rápido que `SELECT * FROM ...` por beeline, que con Hive en modo local se puede quedar colgado.

Uso:
  python3 scripts/exportar_curated.py --origen data/curated_parquet --destino data/curated_export [tabla ...]
Cada subcarpeta de --origen (partidos, estadisticas_equipo, modelado_partidos...) pasa a <tabla>.csv.
"""
import argparse
import pathlib

import pandas as pd


def a_csv(carpeta, destino):
    df = pd.read_parquet(carpeta)                      # lee todas las particiones temporada=AAAA
    if "temporada" in df.columns:
        df["temporada"] = pd.to_numeric(df["temporada"].astype(str), errors="coerce").astype("Int64")
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            df[c] = s.dt.strftime("%Y-%m-%d %H:%M:%S") if (s.dt.hour.fillna(0) + s.dt.minute.fillna(0)).any() \
                else s.dt.strftime("%Y-%m-%d")
        elif s.dtype == "float64" and s.notna().any() and (s.dropna() % 1 == 0).all():
            df[c] = s.astype("Int64")                  # enteros con huecos: 27 y no 27.0
        elif s.dtype == object and s.map(lambda x: hasattr(x, "isoformat")).any():
            df[c] = s.map(lambda x: x.isoformat() if hasattr(x, "isoformat") else x)
    ruta = destino / f"{carpeta.name}.csv"
    df.to_csv(ruta, index=False)
    print(f"{carpeta.name}: {len(df):,} filas x {df.shape[1]} columnas -> {ruta}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--origen", default="data/curated_parquet")
    ap.add_argument("--destino", default="data/curated_export")
    ap.add_argument("tablas", nargs="*", help="si no se indica, todas las subcarpetas")
    a = ap.parse_args()
    origen, destino = pathlib.Path(a.origen), pathlib.Path(a.destino)
    destino.mkdir(parents=True, exist_ok=True)
    for t in (a.tablas or sorted(p.name for p in origen.iterdir() if p.is_dir())):
        a_csv(origen / t, destino)


if __name__ == "__main__":
    main()
