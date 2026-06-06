# T05 · Otras Técnicas de Aprendizaje Supervisado 🔴

> Tema CRÍTICO para el PRÁCTICO: KNN, Decision Tree y Random Forest son ejercicios fijos (2-3 pts). En sklearn son 3-4 líneas cada uno.

## Resumen corto
Además del MLP, el examen pide modelos "clásicos": **KNN** (clasifica por los k vecinos más cercanos), **árboles de decisión** (reglas if/else aprendidas, ID3 usa ganancia de información) y **Random Forest** (conjunto de muchos árboles que votan, reduce varianza). SVM aparece como concepto.

## Resumen completo
### KNN (K-Nearest Neighbors)
- No "entrena": guarda los datos. Para predecir, mira los **k** ejemplos más cercanos (distancia euclídea) y vota la clase mayoritaria.
- **Hiperparámetro k**: pequeño → más varianza (overfit); grande → más sesgo.
- **MUY sensible a la escala** → ¡escalar siempre! Sensible al ruido.
- `KNeighborsClassifier(n_neighbors=k)`.

### Árboles de decisión (ID3 / CART)
- Particiona el espacio con preguntas sobre features. **ID3** elige la feature que maximiza la **ganancia de información** (reduce la **entropía**); CART usa **Gini**.
- Interpretable (reglas claras), robusto al ruido y a escalas. Tiende a **overfit** si crece sin límite → podar / limitar profundidad.
- `DecisionTreeClassifier(random_state=42, max_depth=…)`.

### Random Forest
- **Ensemble** (bagging) de muchos árboles entrenados con muestras/feature aleatorias; predicen por **votación**. Reduce la varianza de un árbol único → suele ser el más preciso "out of the box".
- `RandomForestClassifier(n_estimators=100, random_state=42)`.

### SVM (concepto)
- Busca el hiperplano que **maximiza el margen** entre clases. Con **kernels** modela fronteras no lineales.

## Conceptos clave
- KNN: k, distancia, escalado. Árbol: entropía, ganancia de información, Gini, profundidad, overfitting. Ensemble/bagging, votación. SVM: margen, kernel.

## Preguntas frecuentes
- *"Juego procedural / quiero reglas robustas al ruido, ¿qué modelo?"* → **árboles de decisión** (definen reglas; MLP/KNN son más sensibles al ruido). 📌 (respuesta tipo de Marcos, válida)
- *"¿Cuál suele dar mejor accuracy sin tocar mucho?"* → Random Forest.

## Errores habituales
- ❌ No escalar para KNN (resultados pésimos).
- ❌ Olvidar `random_state` cuando el examen lo fija (resultados no reproducibles → no llegas al umbral pedido).
- Confundir Decision Tree (1 árbol) con Random Forest (muchos).

## Ejercicios recomendados
- `03_PRACTICA/04_modelos_sklearn.md`: entrenar los 3 sobre el mismo dataset y comparar matrices de confusión.
