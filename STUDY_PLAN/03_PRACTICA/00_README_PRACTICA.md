# 03 · PRÁCTICA — Guías de código (la parte que da 8 puntos)

Esta carpeta es tu **arma para el examen práctico**. El examen es siempre el mismo pipeline con un CSV distinto. Si dominas estas plantillas y las llevas al examen, apruebas el práctico.

## Orden de lectura
1. **`01_limpieza_datos.md`** — cargar, limpiar, imputar, OHE, escalar, partir train/test.
2. **`03_usar_mi_MLP.md`** — usar TU `MLP.py` correctamente (one-hot, API real sin las funciones rotas).
3. **`04_modelos_sklearn.md`** — MLPClassifier, KNN, RandomForest, DecisionTree.
4. **`05_MLPRelu_softmax.md`** — la modificación del Ej6 (relu + softmax).
5. **`07_errores_comunes.md`** — checklist de fallos típicos y cómo evitarlos.
6. **`../07_PLANTILLA_EXAMEN/`** — ⭐ el notebook ejecutable (`PLANTILLA_EXAMEN.ipynb`) + módulo `practica5_mlp.py` + `GUIA_PLANTILLA.md`. Es la herramienta final del examen.

## Patrón del examen práctico (memorízalo)
```
Ej1 Limpieza + imputación + transformaciones (justificar markdown)   ~0.5-1 pt
Ej2 Scatter de la distribución de clases (colores)                    ~0.5 pt
Ej3 TU MLP: accuracy + confusión, prueba con varias capas ocultas     2 pt
Ej4 MLPClassifier (sklearn)                                            1 pt
Ej5 KNN                                                                1 pt
Ej5 RandomForest (o DecisionTree)                                      1 pt
Ej5 Análisis train/val + comparar modelos + elegir                    ~0.5 pt
Ej6 MLPRelu (relu/sigmoid + softmax/sigmoid)                           1 pt
```

## Setup (una vez)
```bash
pip install numpy pandas matplotlib scikit-learn scipy
```
Lo que el examen permite llevar: apuntes y material propio. **Lleva tu `MLP.py`, tu `MLPRelu.py` y estas plantillas.**
