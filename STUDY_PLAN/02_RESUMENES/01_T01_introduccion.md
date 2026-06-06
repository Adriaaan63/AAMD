# T01 · Introducción al Aprendizaje Automático

## Resumen corto (1 párrafo)
El aprendizaje automático (ML) construye modelos que aprenden patrones a partir de **datos** en lugar de reglas programadas a mano. Se divide en **supervisado** (datos con etiqueta → predecir), **no supervisado** (sin etiqueta → agrupar/reducir) y **por refuerzo** (aprender de recompensas). El flujo típico: datos → preprocesado → entrenamiento → evaluación → predicción.

## Resumen completo
- **Aprendizaje supervisado:** cada ejemplo tiene entrada X y salida conocida y.
  - **Clasificación:** y es categórica (clases). Ej: dementia/heart/Customer del examen.
  - **Regresión:** y es numérica continua. Ej: predecir precio.
- **Aprendizaje no supervisado:** solo X, sin y. Se buscan estructuras.
  - **Clustering** (K-Means, jerárquico), **reducción de dimensionalidad** (PCA).
- **Aprendizaje por refuerzo:** un agente toma acciones y recibe recompensas; aprende una política. (Baja prioridad.)
- **Conjuntos de datos:** se parten en **train** (entrenar), **validation** (ajustar hiperparámetros), **test** (estimación final imparcial).
- **Minería de datos:** proceso de extraer conocimiento de grandes volúmenes (limpieza, transformación, modelado, evaluación).

## Conceptos clave
- Feature (atributo/variable de entrada), target/label (salida), ejemplo/instancia.
- Hiperparámetro (se fija antes: nº capas, k de KNN, λ) vs parámetro (se aprende: pesos).
- Generalización: rendir bien en datos NO vistos.

## Preguntas frecuentes
- *"Tengo datos sin etiquetar, ¿cómo lo afronto?"* → no supervisado: clustering (K-Means) para agrupar, PCA para reducir dimensión. Si luego consigo etiquetas, paso a supervisado.

## Errores habituales
- Confundir clasificación (clases) con regresión (números).
- Evaluar en los mismos datos de entrenamiento (optimismo engañoso).

## Relación con el examen
Encuadre conceptual. Cae en preguntas tipo "qué enfoque/modelo usar". Ver `00_TEORIA_EXPRESS.md` Familia 3.
