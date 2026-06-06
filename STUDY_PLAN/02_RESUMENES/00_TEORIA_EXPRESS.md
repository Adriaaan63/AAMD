# 00 · TEORÍA EXPRESS — Las preguntas que SIEMPRE caen

> Este es el documento más rentable. La parte teórica (2 pts) se repite cada año con las mismas 6-8 familias de preguntas. Memoriza los **patrones de respuesta** y los puntos clave. Respuestas reales del profesor incluidas (📌).

---

## 🎯 Familia 1 — Diagnóstico por accuracy (train / validation / test / baseline)

La pregunta más frecuente. Te dan números y debes razonar. **Regla mental:**

```
1) Compara TRAIN vs BASELINE:
   - train < baseline  → UNDERFITTING (alto sesgo): el modelo no aprende.
   - train ≥ baseline  → el modelo SÍ es capaz de aprender.
2) Compara TRAIN vs VALIDATION:
   - train alto y val mucho más bajo (gap grande) → OVERFITTING (alta varianza): no generaliza.
   - train ≈ val ≈ alto → generaliza bien ✅
3) TEST muy por debajo de val → el test no se parece a train/val (distribución distinta) o data leakage en la selección.
```

**Acciones según el caso:**

| Situación | Diagnóstico | Qué hacer |
|---|---|---|
| train < baseline | Underfitting / alto sesgo | Modelo **más grande** (más capas/neuronas), más features, menos regularización, entrenar más. |
| train alto, val bajo (gap) | Overfitting / alta varianza | **Más datos**, **más regularización** (λ), data augmentation, modelo más simple, dropout. |
| train≈val pero ambos < baseline | El modelo no da más | Cambiar de modelo / arquitectura, mejores features. |
| val varía mucho con random_state | **Alta varianza** / pocos datos | Más datos, **validación cruzada (k-fold)**, fijar semilla. |

**Ejemplos reales resueltos:**
- 📌 *Train 70%, Val 50%, baseline 65%:* train supera el baseline → el modelo **sí aprende pero no generaliza** (overfitting). Solución: **subir regularización** y **añadir más datos**.
- 📌 *Train 60%, Val 50%, baseline 80%:* ni siquiera train llega al baseline → el modelo **no es capaz de aprender el patrón** (underfitting). Solución: **modelo más grande** (más neuronas/capas) o cambiarlo.
- 📌 *Train 90, Val 80, Test 70:* train y val cerca y por encima de baseline (generaliza razonablemente), pero test bastante más bajo → posibles **errores con datos nuevos** / el test no representa bien la distribución de entrenamiento.
- 📌 *MLP: train 87% supera baseline 80%, val máx 77% probando todos los hiperparámetros:* es **overfitting que no se arregla con hiperparámetros** → hay que actuar sobre los **datos**: conseguir más datos, mejores/más features, o reducir ruido. (No basta tocar el modelo.)

---

## 🎯 Familia 2 — Matriz de confusión (precision / recall)

Te dan 1-2 matrices y debes interpretar. **Ojo a qué son filas y columnas** (lo dicen en el enunciado; a veces filas=predicho, a veces filas=real).

Para clase positiva T (caso "filas = predicho, columnas = real"):
```
            real T   real F
pred T        TP       FP
pred F        FN       TN

Precision = TP / (TP + FP)   → "de lo que predigo positivo, cuánto acierto"
Recall    = TP / (TP + FN)   → "de los positivos reales, cuántos capturo"
Accuracy  = (TP + TN) / total
```

**Cómo razonar la respuesta:**
- Calcula precision y recall de cada modelo.
- Modelo con **muchos FP** (falsos positivos) → baja precisión → malo cuando un falso positivo es caro.
- Modelo con **muchos FN** → bajo recall → malo cuando NO detectar un positivo es caro.
- **Escenarios:**
  - **Recall alto importa** en detección médica (cáncer, cardiopatía): es peor dejar pasar un enfermo (FN) que una falsa alarma.
  - **Precision alta importa** cuando actuar sobre un falso positivo es costoso (p. ej. bloquear a un usuario legítimo, spam que borra correo bueno).

📌 *Respuesta tipo:* "El modelo A genera muchos falsos positivos (baja precisión), el B tiene mayor precisión en los aciertos. A no sirve donde necesitemos alto recall; B es preferible donde necesitemos alta precisión."

---

## 🎯 Familia 3 — ¿Qué modelo/red uso para este escenario?

Mapa de decisión (memorízalo):

| Escenario | Modelo | Por qué |
|---|---|---|
| Clasificar **imágenes** | **CNN** (convolucional) | Capta patrones espaciales; las full-connected solas no escalan bien con píxeles. |
| MLP no funciona con imágenes grandes (128×128) | **Extractor de características** (CNN o **autoencoder**) ANTES del MLP | Reduce dimensionalidad y extrae lo relevante; luego el MLP clasifica. 📌 |
| **Comprimir/reducir memoria** de sprites o imágenes | **Autoencoder** | El codificador comprime a un vector pequeño (espacio latente), el decodificador reconstruye. 📌 |
| **Generar** imágenes nuevas (dígitos, sprites) | **Autoencoder / GAN** | Entrenas a reconstruir; al introducir un código generas la imagen. |
| **Imitar a un jugador** con datos guardados (estado→acción) | **MLP** (clasificación de la acción) | Datos etiquetados estado→acción; salida = nº de acciones posibles. |
| Datos **sin etiquetar** | **No supervisado**: clustering (K-Means), PCA | No hay target; agrupas/reduces. |
| Agente que aprende **sin datos previos**, por prueba y error | **Aprendizaje por refuerzo** (Q-Learning / Deep RL) | No hay dataset; aprende de recompensas. ⚪ (este año no entró) |
| Juego **procedural / reglas claras**, robusto al ruido | **Árboles de decisión** | Aprenden reglas; menos sensibles al ruido que MLP/KNN. 📌 |
| Predecir un **valor continuo** (precio) | **Regresión** (lineal, o combinada con otros) | Target numérico, no clases. |

📌 **Linealidad (pregunta clásica):** ¿qué modelo modela problemas NO lineales?
- Árbol ID3 → modela no lineal por particiones (¡ojo! el profesor lo trata como "lineal" en su respuesta modelo — sigue SU criterio: dijo que ninguno).
- CNN con solo capas convolucionales + Linear final → **lineal** (sin activaciones no lineales).
- MLP con activación `y = m·z + b` → **lineal** (activación lineal ⇒ red lineal).
- 📌 Respuesta del profesor: **"Ninguno"**, porque sin activación no lineal todo queda lineal. Un MLP solo es no lineal **si su activación es no lineal** (sigmoid, relu).

---

## 🎯 Familia 4 — Nº de neuronas de entrada/salida con One-Hot Encoding (OHE)

Patrón: te describen un escenario de juego (casillas, acciones) y debes contar neuronas.

**Reglas:**
- **Salida** = nº de clases/acciones posibles (una neurona por clase, softmax). Si es binario, 1 neurona sigmoid.
- **Entrada**: cada variable **categórica** se expande con OHE = (nº de categorías) neuronas. Variables numéricas = 1 neurona cada una.

📌 **Ejemplo resuelto:** acciones {izq, der, arriba, abajo, disparar, coger} = **6 salidas**. Entrada: 8 casillas adyacentes, cada una toma 6 valores (vacía, suelo, enemigo, mejora, vida, pared) → OHE: 8×6 = 48. Más la posición (2 coords) → **48 + 2 = 50 entradas**.

📌 **Otro:** 4 acciones → 4 salidas; 8 casillas con 4 valores → 32; + posición jugador (1) + posición de N enemigos → **33 + N entradas**.

→ **Siempre:** salidas = nº clases; entradas = Σ(categorías por variable categórica) + variables numéricas.

---

## 🎯 Familia 5 — Pocos datos / mejorar generalización

📌 Estrategias para **maximizar datos escasos**:
- **Data augmentation** (rotar, ruido, recortes en imágenes).
- **Validación cruzada (k-fold)** para aprovechar todos los datos.
- **Regularización** (L2, dropout) para no sobreajustar.
- **Transfer learning** / extractor preentrenado.
- Recolectar más datos / generar sintéticos.

---

## 🎯 Familia 6 — Autoencoder (identificar y explicar)

Si te dan dos `Sequential` donde uno va de grande→pequeño y otro de pequeño→grande (espejo):
- 📌 Es un **autoencoder**. `model01` = **codificador** (comprime, p.ej. 32×32 → 9), `model02` = **decodificador** (reconstruye 9 → 32×32).
- **Para qué sirve:** extractor de características / compresión / reducción de dimensionalidad. El cuello de botella (capa central pequeña) es el **espacio latente**.
- Se entrena para que la **salida reconstruya la entrada** (entrada = objetivo).

---

## 🎯 Familia 7 — CNN como preprocesado del MLP

📌 Si un MLP no clasifica bien imágenes 128×128: antes de clasificar, **extraer características** con una CNN o autoencoder (también vale escalar, pasar a B/N, data augmentation), y luego el MLP clasifica sobre esas características. La idea central que buscan: **extractor de características previo**.

---

## 🎯 Familia 8 — Refuerzo (BAJA prioridad)

⚪ Solo por si acaso: agente sin datos, espacio enorme (10000×10000) → **aprendizaje por refuerzo**. Q-Learning "a pelo" no escala → **discretizar/comprimir estados** o **Deep RL**. Problema clave: diseñar la **recompensa** (recompensas intermedias, no solo al final). 📌 *El profesor avisó: "este año no llegamos a refuerzo, no caería".* No le dediques tiempo.

---

## ✅ Checklist de memorización (deberías poder recitar)
- [ ] Las 4 reglas de diagnóstico (underfit/overfit/varianza/test).
- [ ] Fórmulas precision/recall y cuándo importa cada una.
- [ ] El mapa "escenario → modelo".
- [ ] Cómo contar neuronas con OHE (salidas=clases, entradas=Σcategorías+numéricas).
- [ ] Qué es un autoencoder (codificador/decodificador, latente).
- [ ] "Ninguno es no lineal sin activación no lineal".
