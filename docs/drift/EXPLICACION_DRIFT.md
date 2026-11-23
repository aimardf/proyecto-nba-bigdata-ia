# Drift del objetivo 2 y variable tiempo · HoopMind

## El problema, en sencillo
Los equipos de la NBA anotan cada vez más. En el entrenamiento (2000-2022) un partido tiene de media **203,7 puntos**; desde 2024 son **229,3**. Es como aprender a calcular el precio de un menú con precios de 2005 y usarlo hoy: el modelo no se equivoca por tonto, sino porque el mundo ha cambiado. A eso se le llama **data drift**.

Dos riesgos distintos:
1. **Drift real**: lo que el modelo aprendió ya no vale (el baseline "la media" falla muchísimo: R² = -1,66 en el lote 2).
2. **Extrapolación absurda**: si el modelo usa el año o los puntos absolutos como entrada, un modelo lineal los "continúa" sin freno. Con una liga inventada de +800 puntos, el Ridge actual predice **1.052 puntos** por partido. Un árbol hace lo contrario: no pasa de lo que vio (predice ~235 aunque la liga anotara 300).

## Cómo lo medimos (scripts/drift_y_tiempo.py)
- **PSI** (Population Stability Index): compara la distribución de una variable en el entrenamiento y en lo reciente. <0,1 estable, 0,1-0,25 vigilar, >0,25 drift.
- **KS**: máxima distancia entre las dos distribuciones (0 = iguales, 1 = totalmente distintas).

Resultado (train ≤2022 frente a temporadas ≥2024):
| variable | PSI | KS | veredicto |
|---|---|---|---|
| puntos_totales (objetivo) | 1,53 | 0,46 | DRIFT |
| suma_pts_favor_ult10 / suma_pts_contra_ult10 | 5,2 / 5,3 | 0,73 / 0,72 | DRIFT |
| 13 diferencias local-visitante | ≤0,05 (triples 0,12) | ≤0,10 | estables |
| **mismas variables y objetivo, relativos a la liga** | **≤0,007** | ≤0,03 | **estables** |

Es decir: el drift está concentrado en **el nivel absoluto de puntos**. Las diferencias entre equipos (relativas) no cambian. Ejemplo: "el local anota 3 puntos más que el visitante" significa lo mismo en 2005 que en 2025; "el partido tendrá 205 puntos" no.

## La solución elegida: objetivo relativo a la liga (+ clip)
En vez de predecir "puntos totales", predecimos **cuánto se desvía el partido de la media reciente de la liga** y se la volvemos a sumar:

`puntos_predichos = media_liga_reciente + modelo(entradas relativas)`

- `media_liga_reciente` = media de puntos de los 600 partidos anteriores (≈ mes y medio; se calcula solo con partidos pasados, sin fuga).
- Las dos entradas absolutas (`suma_pts_*`) se reemplazan por `suma - media_liga_reciente`.
- **No se usa año ni temporada como entrada.**
- Se acota la salida a **[127, 319]** puntos: percentiles 0,1 % y 99,9 % de todo el histórico con 10 % de margen. Sirve de red de seguridad contra datos corruptos, no para "arreglar" el modelo.

Ejemplo: la media de liga reciente es 229. El modelo ve que ambos equipos anotan 4 puntos más que la liga: predice +8 → 237. En 2005 con liga 195 habría predicho 203. Es el **mismo modelo** sin reentrenar.

## Evidencia (validación temporal: entrena ≤2022, prueba 2023 y ≥2024)
| estrategia | MAE lote 2 | R² lote 2 | sesgo lote 2 |
|---|---|---|---|
| A actual, Ridge | 15,12 | 0,091 | +0,72 |
| A actual, boosting | 15,32 | 0,063 | -1,57 |
| B + año (con resúmenes), Ridge | 15,11 | 0,091 | +0,29 |
| B2 solo año (sin resúmenes), Ridge | 15,92 | -0,01 | -2,38 |
| C entradas relativas pero objetivo absoluto (error de diseño) | 26,6 | -1,52 | -25 |
| **D objetivo relativo, Ridge** | **15,12** | **0,090** | **+0,06** |
| D objetivo relativo, boosting | 15,18 | 0,085 | +0,07 |
| E ventana de 8 temporadas | 15,11 / 15,42 | 0,091 / 0,059 | +0,16 / +0,29 |
| F pesos por recencia (vida media 3) | 15,10 / 15,22 | 0,093 / 0,081 | +0,06 / +0,26 |

Lectura honesta:
- El error medio apenas cambia (≈15 puntos): el partido de baloncesto es muy ruidoso, el R² es bajo en todas las variantes. El modelo actual ya "seguía" parte del drift porque `suma_pts_*` se calcula con los últimos 10 partidos.
- La ventaja de D está en el **sesgo** (de +0,7 / -1,6 a +0,06) y, sobre todo, en que **no necesita reentrenarse**: un modelo entrenado solo hasta 2012 mantiene MAE 15,40 en 2025 con D, frente a 16,83 con el modelo actual (el reentrenado cada año da 15,42). Ver `modelo_fijo_vs_reentrenado_mae.csv`.
- Ventana reciente y pesos por recencia no mejoran con D (los datos viejos ya no estorban porque todo es relativo). No hace falta complicar el modelo con ellos.
- Ridge ≈ boosting con D; se recomienda Ridge (más simple, y los árboles no extrapolan por encima de lo visto).

## Prueba de estrés "2050" (`prueba_estres_2050.csv`)
Se toma el lote 2 y se cambia el escenario. Predicción media (mín - máx):
| escenario | A Ridge actual | A boosting | solo año Ridge | **D + clip** |
|---|---|---|---|---|
| año = 2050, resto igual | 230 (209-251) | 228 | 270 | **229 (206-252)** |
| liga 2050 siguiendo la tendencia (+67 pts) | 299 | 235 | 270 | **296 (273-319)** |
| liga imposible (+800) | **1.052** | 235 | 270 | **319 (tope)** |

Cómo leerlo:
- Cambiar solo el año a 2050 **no mueve** el modelo D (el año no es entrada): cumple lo que pidió el profe.
- Si la liga realmente subiera (tendencia 2010-2025: +2,7 pts/temporada, +67 en 2050), D sigue la liga con datos reales (296). El boosting se quedaría en 235 (se equivoca por -60), y por eso no es una buena opción.
- Si los datos son absurdos (+800), el Ridge actual dice 1.052 y D sin clip también diría ~1.030; **el clip lo deja en 319**. Honestidad: el clip es una red de seguridad, no inteligencia; si hay un fallo de datos, la monitorización lo debe detectar.

## Mitigación operativa
- **Monitorizar** PSI del objetivo y de `suma_pts_*` por temporada/mes frente al entrenamiento (`drift_por_temporada.csv`); alerta si PSI > 0,25.
- **Reentrenar** al menos una vez por temporada. Con D el modelo envejece mucho mejor, pero el reentreno sigue siendo la práctica correcta.
- Registrar en MLflow cada reentreno (experimento `hoopmind_obj2_drift`).

## Ficheros
`scripts/drift_y_tiempo.py` (todo el análisis) · `docs/drift/*.csv` y `drift_objetivo2.png` · MLflow: `sqlite:///mlflow.db`, experimento `hoopmind_obj2_drift` (16 estrategias + estrés + resumen).
