# 04 · Análisis de Exámenes

Análisis de 4 exámenes (2024, 2025, 2026 + teóricos resueltos). **Conclusión principal: el examen es casi idéntico cada año.** Cambia el dataset, no la estructura.

---

## 1. Tabla comparativa de exámenes prácticos

| Año | Dataset | Tipo target | Umbral accuracy | random_state / test | Modelos pedidos |
|---|---|---|---|---|---|
| 2024 (enero) | dementia | 3 clases (Demented/Nondemented/Converted) | MLP ≥65%, Tree ≥85% | (Tree rs=42) | MLP propio, DecisionTree, + MLP binario |
| 2025 (enero) | heart.csv | binaria (cardiopatía) | ≥84% todos | rs=0, test 25% | MLP propio, MLPClassifier, KNN, RandomForest |
| 2026 (enero, **suspendido**) | Customer.csv | 4 clases (segmentación) | MLP ≥50%, KNN ≥48%, RF ≥53% | rs=13, test 20% | MLP propio, MLPClassifier, KNN, RandomForest |

> El umbral de 2026 era **bajísimo** (50%) → era un examen "fácil" de aprobar en práctica. El suspenso vino por problemas de ejecución/pipeline, no por dificultad del modelo.

## 2. Estructura práctica recurrente (frecuencia de aparición)

| Ejercicio | 2024 | 2025 | 2026 | Frecuencia |
|---|:---:|:---:|:---:|---|
| Limpieza + transformaciones (+imputación) | ✅ | ✅ | ✅ | **100%** |
| Scatter distribución de clases | ✅ | ✅ | ✅ | **100%** |
| MLP propio + accuracy + confusión (prueba multicapa) | ✅ | ✅ | ✅ | **100%** |
| MLPClassifier sklearn | — | ✅ | ✅ | 67% |
| KNN | — | ✅ | ✅ | 67% |
| RandomForest | — | ✅ | ✅ | 67% |
| DecisionTree | ✅ | — | — | 33% |
| Análisis train/val + comparar/elegir modelo | ✅ | ✅ | ✅ | **100%** |
| Modificar MLP (relu/softmax/binario) | ✅ | ✅ | ✅ | **100%** |

→ **Apuesta segura:** limpieza, scatter, MLP propio, 2-3 modelos sklearn, comparación, modificación del MLP. Prepara todo eso = `07_PLANTILLA_EXAMEN/`.

## 3. Teórico: familias de preguntas y frecuencia

| Familia | Aparece en | Frec. |
|---|---|---|
| Diagnóstico accuracy train/val/test vs baseline (overfit/underfit) | 2024, 2025, 2026, junio | **muy alta** |
| Matriz de confusión (precision/recall, qué modelo) | 2025, 2026 | alta |
| Qué modelo/red para escenario X (CNN, autoencoder, MLP, árbol, refuerzo, regresión) | 2024, 2025, 2026, junio | **muy alta** |
| Nº neuronas entrada/salida con OHE | 2024, junio | alta |
| Linealidad de modelos | 2024, junio | media |
| Autoencoder (identificar/explicar) | junio | media |
| Pocos datos / data augmentation | 2025 | media |
| Extractor de características antes del MLP (imágenes) | 2024, junio | media |
| Refuerzo | junio (con nota "no entró") | baja ⚪ |

→ Todas resueltas con plantilla en `02_RESUMENES/00_TEORIA_EXPRESS.md`.

## 4. Tendencias y predicción para la extraordinaria
- **Casi seguro** mismo formato: 1 CSV de clasificación + pipeline de 6-9 ejercicios + 4-8 teóricas cortas.
- Dataset nuevo (médico o de videojuegos), pero el código se reutiliza al 90%.
- Modificación del MLP: probablemente relu/softmax otra vez (o binario). Lleva `MLPRelu.py` preparado.
- Teórico: seguro caen diagnóstico por accuracy y "qué modelo usar".

## 5. Datasets de práctica disponibles en el repo
- `Practicas resueltas/Clustering/data/shopping_data.csv` (clustering / scatter).
- `Examenes/.../examen_24/dementia_dataset.csv` (el de 2024, úsalo para simulacro real).
- `Examenes/Practico/.../ExamenEnero20024/dementia_dataset.csv` (idem).

> Recomendación: usa **dementia_dataset.csv** para el simulacro intermedio y, si consigues heart.csv/Customer.csv, mejor aún.
