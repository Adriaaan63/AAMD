# 00 · Mapa de la Asignatura y Diagnóstico

**Asignatura:** Aprendizaje Automático y Minería de Datos (AAMD)
**Grado:** Desarrollo de Videojuegos (UCM)
**Objetivo:** Aprobar la **convocatoria extraordinaria** (≥ 5/10) antes del 15 de junio de 2026.
**Punto de partida:** Ordinaria suspendida → **Práctico 1,25/8 · Teórico 0,65/2 = ~1,9/10**.

---

## 1. Estructura del examen (INVARIANTE 2024 → 2026)

| Parte | Peso | Tiempo | Formato |
|---|---|---|---|
| **Teórica** | 2 pts | 30–50 min | 4–8 preguntas cortas razonadas. Documento de texto/PDF. |
| **Práctica** | 8 pts | ~2–2.5 h | Jupyter Notebook (.ipynb) en zip + tu librería MLP. Debe **ejecutar sin intervención**. |

> ⚠️ **Regla crítica:** la librería MLP debe basarse en la de tus prácticas 4/5. Si es radicalmente distinta = **copia = suspenso**. La tuya ya vale, no la cambies de raíz.
> ⚠️ Si el notebook **no ejecuta**, hay penalización (hasta −1). Rutas relativas, datos incluidos en el zip.

---

## 2. Temario y prioridad para APROBAR

Prioridad = (peso en examen) × (frecuencia histórica). Escala 🔴 crítico · 🟠 alto · 🟡 medio · ⚪ bajo.

| Tema | Contenido | Dónde cae | Prioridad |
|---|---|---|---|
| **T04 Diseño de sistemas ML** | train/val/test, baseline, overfitting/underfitting, sesgo-varianza, matriz confusión, precision/recall, regularización | **Teórico (casi todas las preguntas)** | 🔴 |
| **T03 Redes de neuronas (MLP)** | perceptrón, feedforward, backprop, sigmoid, OHE entrada/salida | **Práctico Ej3+Ej6 (3pts) + Teórico** | 🔴 |
| **T05 Otras técnicas supervisadas** | KNN, árboles de decisión (ID3), Random Forest, SVM | **Práctico Ej4-5 (2-3pts) + Teórico (linealidad)** | 🔴 |
| **T01 Introducción** | tipos de aprendizaje, supervisado vs no supervisado | Teórico (encuadre) | 🟠 |
| **T07 Deep Learning** | CNN (imágenes), autoencoders (compresión/extracción) | **Teórico (recurrente)** | 🟠 |
| **T02 Regresión y clasificación** | regresión lineal/logística, función coste | Teórico (linealidad, precios) | 🟡 |
| **T06 No supervisado** | clustering, K-Means, jerárquico, PCA | Teórico (datos sin etiquetar) | 🟡 |
| **T09 IA Generativa** | GAN, generación de imágenes | Teórico (generar dígitos/sprites) | 🟡 |
| **T08 Aprendizaje por refuerzo** | Q-Learning, recompensas | Teórico (agente sin datos) — *profesor: "no entró este año"* | ⚪ |

**Conclusión:** el 80% de los puntos se juega en **T03 + T04 + T05**. Es donde hay que invertir el tiempo.

---

## 3. El pipeline práctico SIEMPRE es el mismo

Todos los exámenes prácticos siguen este guion con un CSV de clasificación distinto (dementia 2024, heart 2025, Customer 2026):

```
1. Cargar y LIMPIAR el dataset (nulos, columnas inútiles, IDs)   → 0.5-1 pt
2. IMPUTAR algún atributo + OHE categóricas + escalar             → (dentro de Ej1)
3. VISUALIZAR distribución de clases (scatter coloreado)          → 0.5 pt
4. Tu MLP propio: accuracy + matriz confusión, ≥1 prueba ≥3 capas → 2 pts
5. MLPClassifier de sklearn                                       → 1 pt
6. KNN                                                            → 1 pt
7. RandomForestClassifier (o DecisionTree)                        → 1 pt
8. Analizar train vs val + comparar modelos + elegir              → 0.5 pt
9. Modificar tu MLP → MLPRelu (relu + softmax, coste con xlogy)   → 1 pt
```

→ **Es mecánico y repetible.** Quien lleva una plantilla que ejecuta, aprueba el práctico. Ver `03_PRACTICA/`.

---

## 4. Diagnóstico: ¿por qué se suspendió la ordinaria? (1,25/8)

Analizado el código entregado (`Examenes/Ordinaria/Practico/mlp.py` + `Utils.py` + `data_mining.py`):

| Hallazgo | Impacto | Solución |
|---|---|---|
| El **núcleo del MLP es correcto** (constructor, feedforward, backprop, coste+L2). | ✅ La base sirve, no reescribir. | Mantener `models/MLP.py`. |
| Funciones helper finales **rotas**: `feedforward(X)[2]` → `IndexError` (solo devuelve `As, Zs`); `compute_gradients` devuelve 2 valores pero `target_gradient`/`costNN` esperan 3 → `ValueError`. | ⚠️ Si el notebook llamaba a estas, **petaba y no ejecutaba** (penalización + 0 en ejercicios). | Usar el MLP por su API real: `feedforward(X)[0][-1]` para la salida. Ver `03_PRACTICA`. |
| Su `Utils.py` del examen era de **exportación ONNX a Unity**; no tenía helpers tabulares. | ⚠️ Sin funciones de imputación/OHE/split/confusión → pipeline improvisado y frágil. | Llevar plantilla de pipeline lista (`03_PRACTICA/02_pipeline_sklearn.md`). |
| `data_mining.py` era del proyecto Unity (tanques), no adaptado a CSV con categóricas. | ⚠️ No reutilizable en el examen tabular. | Plantilla genérica de limpieza+OHE. |
| Probable: target multiclase NO codificado en **one-hot** para el MLP propio (el coste y backprop lo requieren). | 🔴 Causa típica de accuracy ≈ azar. | `pd.get_dummies(y)` / one-hot manual antes de entrenar el MLP. |

**Resumen del diagnóstico:** el problema NO fue entender el MLP, fue **el andamiaje** (limpieza, codificación, conexión de datos con el modelo, notebook ejecutable). Eso se arregla con plantillas y práctica mecánica → es la vía más rápida a ≥5.

---

## 5. Estrategia para aprobar (mínimo esfuerzo, máxima nota)

1. **Asegurar el práctico (8 pts).** Llevar plantilla que ejecute de principio a fin. Solo con limpieza + scatter + MLP funcional + 3 modelos sklearn + análisis ya se superan los 5/8. → con eso solo casi se aprueba.
2. **Teórico (2 pts) es "gratis":** las preguntas se repiten (diagnóstico baseline/overfit, matriz confusión, qué modelo usar, neuronas con OHE). Memorizar los patrones de `02_RESUMENES/00_TEORIA_EXPRESS.md`.
3. **No perder tiempo** en T08 refuerzo ni en demostraciones matemáticas profundas.

**Meta realista:** Práctico 5-6/8 + Teórico 1.2-1.6/2 → **6.5-7.5/10**. Sobradamente aprobado.
