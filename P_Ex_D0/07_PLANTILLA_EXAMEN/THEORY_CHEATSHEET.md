# THEORY CHEATSHEET · AAMD (consulta relámpago en examen)

> Densidad máxima, cero relleno. Para el **desarrollo** ver `../02_RESUMENES/00_TEORIA_EXPRESS.md`. Aquí solo: gatillos → respuesta.

---

## 1 · Diagnóstico (train / val / test / baseline)
| Síntoma | Dx | Acción (palabras clave a escribir) |
|---|---|---|
| train < baseline | **underfit / alto sesgo** | red más grande · +features · −λ · +iteraciones |
| train alto, val mucho < (gap) | **overfit / alta varianza** | +datos · +λ · dropout · modelo más simple · augmentation |
| train ≈ val, ambos < baseline | modelo tope | cambiar arquitectura/modelo · mejores features |
| val cambia mucho con seed | **alta varianza / pocos datos** | k-fold · +datos · fijar semilla |
| test ≪ val | test ≠ distribución train · leakage | revisar split / datos nuevos |
| overfit que NO baja tocando hiperparámetros | problema de **datos** | +datos · +features · −ruido (no basta el modelo) |

Orden mental: **(1)** train vs baseline → ¿aprende? **(2)** train vs val → ¿generaliza? **(3)** test → ¿representa?

---

## 2 · Matriz de confusión
```
            real +   real −
pred +        TP       FP
pred −        FN       TN
```
- **Precision** = TP/(TP+FP) — "de lo que digo +, cuánto acierto" → sube si pocos **FP**.
- **Recall** = TP/(TP+FN) — "de los + reales, cuántos pillo" → sube si pocos **FN**.
- **Accuracy** = (TP+TN)/total · **F1** = 2·P·R/(P+R).
- **Recall manda** → médico (no dejar pasar enfermo). **Precision manda** → coste alto de falsa alarma (banear usuario, borrar correo bueno).
- ⚠️ Mira en el enunciado si filas=predicho o filas=real (cambia el cálculo).

---

## 3 · Escenario → modelo
| Escenario | Modelo |
|---|---|
| Imágenes (clasificar) | **CNN** |
| MLP no escala con imágenes grandes | **extractor previo** (CNN/autoencoder) → MLP |
| Comprimir/reducir dimensión | **autoencoder** (latente = cuello) |
| Generar imágenes nuevas | **autoencoder / GAN** |
| Imitar jugador (estado→acción etiquetado) | **MLP** clasificación |
| Datos sin etiqueta | **no supervisado**: K-Means, PCA |
| Agente sin datos, prueba/error | **refuerzo** (Q-Learning/Deep RL) ⚪ no entró |
| Reglas claras, robusto a ruido | **árbol de decisión** |
| Predecir valor continuo | **regresión** |

**No-linealidad (📌 profesor):** sin activación no lineal → todo es lineal ⇒ "**ninguno** es no lineal". MLP es no lineal **solo** con sigmoid/relu. CNN solo conv+Linear = lineal.

---

## 4 · Contar neuronas (OHE)
- **Salida** = nº clases (1 por clase, softmax). Binario → **1** (sigmoid).
- **Entrada** = Σ(categorías de cada variable categórica) + (variables numéricas × 1).
- Ej: 8 casillas × 6 valores = 48 + 2 coords = **50 entradas**; 6 acciones = **6 salidas**.

---

## 5 · Activaciones / coste (chuleta MLP)
| | Oculta | Derivada oculta | Salida | Coste |
|---|---|---|---|---|
| sigmoid | 1/(1+e⁻ᶻ) | a·(1−a) | sigmoid | −[y·log ŷ+(1−y)·log(1−ŷ)] |
| relu+softmax | max(0,z) | (a>0)·1 | softmax(z−max) | −xlogy(y,ŷ) |
- δ salida = **a − y** en ambos (softmax+CE y sigmoid+CE). Backprop última capa igual.
- relu: usa **alpha pequeño** (0.1–0.3); `scipy.special.xlogy` evita log(0)=NaN.
- L2: coste += (λ/2m)·Σθ² (sin la columna de bias).

---

## 6 · Pipeline tabular (orden fijo)
`cargar → drop IDs/constantes → imputar (num=mediana, cat=moda) → OHE features + LabelEncoder target → split(stratify) → StandardScaler(fit solo en train) → modelo → accuracy+confusión → análisis`
- Escalar **siempre** para MLP/KNN; árboles/RF no lo necesitan.
- ⚠️ columna muy correlada con el target (ej. CDR↔demencia) = posible "trampa" → coméntalo.

---

## 7 · Modelos sklearn (defaults útiles)
| Modelo | Línea | Nota |
|---|---|---|
| MLPClassifier | `MLPClassifier(hidden_layer_sizes=(64,32), max_iter=1000)` | escalar antes |
| KNN | `KNeighborsClassifier(n_neighbors=k)` | probar k impar; escalar |
| RandomForest | `RandomForestClassifier(n_estimators=200, random_state=42)` | robusto, sin escalar |
| DecisionTree | `DecisionTreeClassifier(random_state=42)` | interpretable, fija seed |

---

## 8 · No supervisado / extra
- **K-Means**: k clusters, minimiza distancia al centroide; elegir k por **codo/inercia**. Escalar antes.
- **PCA**: reduce dimensión proyectando a componentes de máxima varianza; usado para scatter 2D.
- **Autoencoder** = codificador (grande→pequeño) + decodificador (pequeño→grande); entrena salida≈entrada; latente = features comprimidas.

---

## 9 · Mejorar con pocos datos
augmentation · k-fold · regularización (L2/dropout) · transfer learning / extractor preentrenado · recolectar/sintetizar más.

---

## ⚡ Frases-plantilla para markdown del examen
- *Limpieza:* "Elimino IDs (no informativos) y columnas constantes; imputo numéricas con mediana (robusta a outliers) y categóricas con moda; aplico OHE a categóricas."
- *Comparación:* "El modelo X tiene mayor accuracy/recall; en este dominio priorizo **recall** (no dejar pasar positivos) / **precisión** (coste de falsa alarma), luego elijo X."
- *Overfit:* "train≫val indica sobreajuste; aplico más regularización y/o más datos antes de tocar la arquitectura."
- *Refuerzo* ⚪ baja prioridad (profesor: no entró este año).
