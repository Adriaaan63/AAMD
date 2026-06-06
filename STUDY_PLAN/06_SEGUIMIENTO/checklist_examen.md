# Checklist del día del examen (imprimir / tener al lado)

## Antes de empezar
- [ ] Tengo a mano: `MLP.py`, `MLPRelu.py`, carpeta `03_PRACTICA/`, `02_RESUMENES/00_TEORIA_EXPRESS.md`.
- [ ] Nombre en la 1ª celda del notebook y en el nombre del zip.

## Teórico (haz primero lo que sepas seguro)
- [ ] Diagnóstico: comparo train vs baseline (underfit) y train vs val (overfit).
- [ ] Matriz confusión: calculo precision y recall, digo en qué escenario va cada modelo.
- [ ] "Qué modelo": uso el mapa escenario→modelo.
- [ ] Neuronas: salidas=clases; entradas=Σcategorías(OHE)+numéricas.
- [ ] Respondo razonando, aunque sea breve. Toda pregunta contestada suma.

## Práctico (en orden, que SIEMPRE ejecute)
1. [ ] Cargar CSV (ruta relativa) + explorar (`info`, `isna().sum`, `value_counts`).
2. [ ] Limpiar: quitar IDs, **imputar** ≥1 atributo, **justificar en markdown**.
3. [ ] OHE categóricas + LabelEncoder target.
4. [ ] Split con el **random_state y test_size EXACTOS** del enunciado + `StandardScaler`.
5. [ ] Scatter de clases (PCA 2D).
6. [ ] Mi MLP: **one-hot del target**, prueba con ≥3 capas, accuracy + confusión, decir modelo final.
7. [ ] MLPClassifier, KNN, RandomForest: accuracy + confusión cada uno.
8. [ ] Comparar + elegir en markdown (train vs test → over/underfit).
9. [ ] MLPRelu (relu+softmax) o lo que pida el Ej6.

## Rúbrica orientativa (8 pts práctico)
| Ejercicio | Puntos | Conseguido |
|---|---|---|
| Limpieza + imputación justificada | ~1 | ⬜ |
| Scatter | ~0.5 | ⬜ |
| MLP propio + confusión + multicapa | 2 | ⬜ |
| MLPClassifier | 1 | ⬜ |
| KNN | 1 | ⬜ |
| RandomForest/DecisionTree | 1 | ⬜ |
| Análisis/comparación | ~0.5 | ⬜ |
| MLPRelu/modificación | 1 | ⬜ |

## Últimos 10 minutos
- [ ] **Kernel → Restart & Run All**: ejecuta limpio de arriba a abajo.
- [ ] CSV + librerías `.py` dentro del zip. Rutas relativas.
- [ ] Subir teórico (PDF/txt con nombre) y práctico (zip con nombre).

## Mantra
> "Primero que EJECUTE, luego que sea bueno." Un notebook que corre y hace medio examen aprueba; uno perfecto que peta, no.
