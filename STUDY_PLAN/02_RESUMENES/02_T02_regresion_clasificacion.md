# T02 · Regresión y Clasificación

## Resumen corto
La **regresión lineal** predice un valor continuo como combinación lineal de las features (`y = θ₀ + θ₁x₁ + …`). La **regresión logística** adapta esa idea a **clasificación** pasando la salida por una **sigmoide** para obtener una probabilidad entre 0 y 1. Ambas se entrenan minimizando una **función de coste** por **descenso de gradiente**.

## Resumen completo
- **Regresión lineal:** hipótesis `h(x)=θᵀx`. Coste = Error Cuadrático Medio (MSE). Se minimiza con gradiente.
- **Regresión logística (clasificación binaria):** `h(x)=sigmoid(θᵀx)`, sigmoid(z)=1/(1+e^−z). Coste = **entropía cruzada** (log loss). Frontera de decisión en h(x)=0.5.
- **Multiclase:** one-vs-all o softmax (una probabilidad por clase, suman 1).
- **Regularización (L2):** se añade `λ·Σθ²` al coste para penalizar pesos grandes y reducir overfitting. No se regulariza el bias.
- **Descenso de gradiente:** `θ := θ − α·∂J/∂θ`. α = learning rate.
- **Normalización/escalado:** acelera la convergencia y es necesaria cuando las features tienen escalas distintas.

## Conceptos clave
- Sigmoide, entropía cruzada, frontera de decisión, learning rate α, regularización λ.
- Lineal vs no lineal: un modelo es **no lineal solo si su activación es no lineal**.

## Preguntas frecuentes
- *"¿Qué modelo para predecir precios?"* → **regresión** (target continuo); se pueden combinar modelos.
- *"¿Cuál modela problemas no lineales?"* → ver `00_TEORIA_EXPRESS` Familia 3 (respuesta: ninguno sin activación no lineal).

## Errores habituales
- Usar regresión lineal para clasificación (mejor logística).
- Olvidar escalar antes de entrenar.

## Relación con el examen
Base conceptual del MLP (cada neurona es una regresión logística). Cae en "linealidad" y "precios".
