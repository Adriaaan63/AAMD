# 01 · Plan General de Estudio (9 días)

**Hoy:** sábado 6 junio 2026 · **Examen:** ~15 junio 2026 · **Días disponibles:** 9
**Filosofía:** primero asegurar el PRÁCTICO (8 pts, mecánico) → luego TEÓRICO (2 pts, repetitivo). De lo más importante a lo menos. De cero a aprobar.

> Carga sugerida: 3–4 h/día. Si tienes menos tiempo, prioriza los días marcados 🔴.

---

## Vista rápida

| Día | Fecha | Foco | Entregable mental |
|---|---|---|---|
| 1 🔴 | Sáb 6 | Fundamentos + montar entorno + entender el pipeline | Notebook que carga un CSV y lo limpia |
| 2 🔴 | Dom 7 | MLP propio: entenderlo y usarlo bien (OHE, train) | MLP entrena y predice sobre un CSV |
| 3 🔴 | Lun 8 | Modelos sklearn (MLPClassifier, KNN, RandomForest) | Los 3 modelos dan accuracy y matriz confusión |
| 4 🟠 | Mar 9 | Pipeline completo de punta a punta (Simulacro básico) | Notebook completo ejecutable |
| 5 🟠 | Mié 10 | MLPRelu (relu + softmax) + análisis train/val | Ej6 resuelto |
| 6 🔴 | Jue 11 | TEORÍA express: diagnóstico, matrices, qué modelo usar | Respondes las 8 preguntas tipo de memoria |
| 7 🟠 | Vie 12 | Simulacro intermedio (dataset nuevo) cronometrado | Práctico en 2h |
| 8 🔴 | Sáb 13 | Simulacro COMPLETO (examen real) + corregir fallos | Examen entero en tiempo |
| 9 🟡 | Dom 14 | Repaso de errores + memorizar teoría + preparar zip | Todo listo y memorizado |
| — | Lun 15 | **EXAMEN** | Aprobar ✅ |

---

## Detalle diario

### Día 1 (Sáb 6) 🔴 — Fundamentos y entorno
- **Teoría (1h):** leer `02_RESUMENES/01_T01_introduccion.md` y `02_RESUMENES/04_T04_diseno_diagnostico.md`. Entender: qué es aprendizaje supervisado/no supervisado, train/val/test, qué es overfitting/underfitting y baseline.
- **Práctica (2h):**
  - Montar entorno: `pip install numpy pandas matplotlib scikit-learn scipy`.
  - Leer `03_PRACTICA/01_limpieza_datos.md`.
  - Coger `Practicas resueltas/Clustering/data/shopping_data.csv` (o el dataset Customer si lo tienes) y practicar: cargar con pandas, ver `.info()`, `.isna().sum()`, eliminar IDs, imputar nulos, OHE de categóricas, escalar.
- **Meta:** un notebook que lee un CSV y lo deja "limpio y numérico".

### Día 2 (Dom 7) 🔴 — Tu MLP, dominado
- **Teoría (1h):** `02_RESUMENES/03_T03_redes_neuronas.md`. Entender feedforward, backprop a nivel intuitivo, por qué la salida va en one-hot.
- **Práctica (2.5h):**
  - Leer `03_PRACTICA/03_usar_mi_MLP.md` (API real de tu `MLP.py`, sin las funciones rotas).
  - Entrenar tu MLP sobre un CSV limpio: one-hot del target, `MLP([n_in],[capas],[n_clases])`, `backpropagation()`, `predict(feedforward(X)[0][-1])`, accuracy + matriz confusión.
  - Probar con **3 capas ocultas** (lo pide el examen).
- **Meta:** tu MLP entrena y supera el azar en un dataset real.

### Día 3 (Lun 8) 🔴 — Modelos de sklearn
- **Teoría (1h):** `02_RESUMENES/05_T05_supervisado_otros.md` (KNN, árboles, Random Forest) + `02_RESUMENES/07_T07_deeplearning.md` (CNN/autoencoder, para teoría).
- **Práctica (2h):** `03_PRACTICA/04_modelos_sklearn.md`. Entrenar `MLPClassifier`, `KNeighborsClassifier`, `RandomForestClassifier`: cada uno con accuracy + `confusion_matrix`. Probar a subir accuracy ajustando hiperparámetros sencillos.
- **Meta:** los 3 modelos funcionan en 10 líneas cada uno.

### Día 4 (Mar 9) 🟠 — Pipeline completo
- **Práctica (3h):** hacer el **Simulacro Básico** entero (`05_SIMULACROS/01_basico.md`) de principio a fin en un solo notebook, sin mirar soluciones. Luego corregir con la solución.
- **Meta:** notebook completo que ejecuta de arriba a abajo sin errores.

### Día 5 (Mié 10) 🟠 — MLPRelu + análisis
- **Práctica (2.5h):** `03_PRACTICA/05_MLPRelu_softmax.md`. Crear copia de tu MLP con `function="relu"/"sigmoid"` y `out_function="softmax"/"sigmoid"`. Derivada relu = `(a>0)*1`. Coste con `scipy.special.xlogy`.
- **Teoría (0.5h):** análisis train vs validation (cómo redactar la conclusión del Ej de análisis).
- **Meta:** MLPRelu entrena con relu+softmax y da resultado similar a sigmoid.

### Día 6 (Jue 11) 🔴 — Teoría express
- **Teoría (3h):** `02_RESUMENES/00_TEORIA_EXPRESS.md`. Memorizar las **plantillas de respuesta** para los 8 tipos de pregunta. Hacer las FAQ. Resolver de memoria el `05_SIMULACROS/teorico_tipo.md`.
- **Meta:** responder cualquier pregunta de diagnóstico/matriz/modelo en <3 min.

### Día 7 (Vie 12) 🟠 — Simulacro intermedio cronometrado
- **Práctica (2.5h):** `05_SIMULACROS/02_intermedio.md` con dataset nuevo, **cronómetro a 2h**. Sin soluciones delante.
- **Meta:** terminar el práctico en tiempo.

### Día 8 (Sáb 13) 🔴 — Simulacro COMPLETO (modo examen)
- **(3h):** `05_SIMULACROS/03_completo.md` = examen entero (teórico 50 min + práctico 2h). Condiciones reales.
- Corregir con soluciones. Anotar fallos en `06_SEGUIMIENTO/registro_errores.md`.
- **Meta:** sacar ≥5/10 en el simulacro.

### Día 9 (Dom 14) 🟡 — Repaso final y logística
- Repasar `06_SEGUIMIENTO/registro_errores.md` (solo tus fallos).
- Releer `00_TEORIA_EXPRESS.md` una vez.
- **Preparar el material que llevas al examen** (permitido: apuntes, campus): tener a mano `03_PRACTICA/` y tu `MLP.py` limpio + `MLPRelu.py`. Verificar que tu plantilla de notebook **ejecuta** en frío.
- Dormir. No estudiar nada nuevo.

---

## Reglas de oro durante el examen
1. **Primero haz que ejecute** (aunque sea mal), luego mejora. Un notebook que corre puntúa; uno que peta, no.
2. Rutas **relativas** al CSV, datos en el mismo zip.
3. Target multiclase → **one-hot** para tu MLP; etiqueta entera para sklearn.
4. **Escala** los datos (StandardScaler) antes de MLP y KNN.
5. Si no llegas al accuracy mínimo, **igual entrégalo**: limpieza+scatter+modelos puntúan aunque no llegues al umbral.
6. Justifica TODO en celdas markdown (limpieza, imputación, elección de modelo): da puntos fáciles.
7. Reserva 10 min finales para **"Reiniciar kernel y ejecutar todo"** y comprobar que corre limpio.
