# T04 · Diseño de Sistemas de Aprendizaje (Diagnóstico) 🔴

> Tema CRÍTICO para el TEÓRICO. Casi todas las preguntas teóricas salen de aquí. Estúdialo junto a `00_TEORIA_EXPRESS.md`.

## Resumen corto
Para construir un buen sistema de ML hay que: partir los datos (train/val/test), fijar un **baseline**, diagnosticar si el modelo sufre **sesgo alto (underfitting)** o **varianza alta (overfitting)**, y actuar en consecuencia. Las métricas (accuracy, precision, recall, matriz de confusión) guían las decisiones.

## Resumen completo
### Partición de datos
- **Train**: entrenar parámetros. **Validation**: ajustar hiperparámetros y diagnosticar. **Test**: estimación final, no se toca durante el entrenamiento.
- `random_state` fija la partición (reproducibilidad). Si los resultados **varían mucho** según `random_state` → **alta varianza / pocos datos** → usar **validación cruzada (k-fold)**.

### Baseline
Nivel de rendimiento de referencia (humano, modelo simple, requisito). Se compara contra train/val para diagnosticar.

### Sesgo vs Varianza
- **Sesgo alto (underfitting):** train por debajo del baseline. El modelo es demasiado simple. → más capacidad, más features, menos regularización.
- **Varianza alta (overfitting):** train alto, val mucho más bajo. No generaliza. → más datos, más regularización, simplificar, data augmentation.

### Métricas y matriz de confusión
```
Accuracy  = aciertos / total
Precision = TP / (TP+FP)
Recall    = TP / (TP+FN)
F1        = 2·P·R/(P+R)
```
- Accuracy engaña con **clases desbalanceadas** → mirar precision/recall/F1.
- Elegir métrica según el coste de cada tipo de error (FP vs FN).

### Regularización
Penaliza pesos grandes (L2: `λΣθ²`) para reducir overfitting. λ grande → más simple (puede underfit); λ pequeño → más flexible (puede overfit).

## Conceptos clave
- train/val/test, baseline, sesgo, varianza, overfitting, underfitting, regularización λ, validación cruzada, matriz de confusión, precision, recall, F1, desbalanceo.

## Preguntas frecuentes
TODAS las de `00_TEORIA_EXPRESS.md` Familias 1, 2 y 5. Memoriza esas plantillas.

## Errores habituales
- Diagnosticar overfitting cuando en realidad es underfitting (train < baseline).
- Dar como "buena" una accuracy alta en dataset desbalanceado.
- Tocar el test set para decidir hiperparámetros (eso es trampa metodológica).

## Relación con el examen
Es el 70-90% del teórico. También el Ej de "análisis train vs validation" del práctico (0.5 pt).
