#!/usr/bin/env python3
"""
ddl_desde_cabecera.py · HoopMind

Escribe el CREATE EXTERNAL TABLE de Hive a partir de la cabecera real de un CSV,
con todas las columnas como STRING (la limpieza convierte los tipos despues).
Asi el DDL no depende de copiar a mano 59 o 104 nombres de columna.

Uso:
  python3 scripts/ddl_desde_cabecera.py <csv_local> <tabla_hive> <ruta_hdfs>
  python3 scripts/ddl_desde_cabecera.py data/dirty/Games.csv nba_dirty.games /nba/dirty/games
"""
import csv
import re
import sys

if len(sys.argv) != 4:
    sys.exit(__doc__)
ruta, tabla, ubicacion = sys.argv[1:]
with open(ruta, newline="", encoding="utf-8") as f:
    cabecera = next(csv.reader(f))

malas = [c for c in cabecera if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", c)]
if malas:
    sys.exit(f"ERROR: nombres de columna no validos para Hive: {malas}")

cols = ",\n  ".join(f"`{c}` STRING" for c in cabecera)
print(f"""DROP TABLE IF EXISTS {tabla};
CREATE EXTERNAL TABLE {tabla} (
  {cols}
)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
WITH SERDEPROPERTIES ("separatorChar" = ",", "quoteChar" = "\\"", "escapeChar" = "\\\\")
STORED AS TEXTFILE
LOCATION '{ubicacion}'
TBLPROPERTIES ("skip.header.line.count" = "1");
""")
