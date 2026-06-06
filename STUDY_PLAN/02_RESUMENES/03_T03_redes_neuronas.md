# T03 · Redes de Neuronas (Perceptrón Multicapa) 🔴

> Tema CRÍTICO: es el corazón del práctico (Ej3 + Ej6 = 3 pts) y cae en teoría. Tu propia librería `MLP.py` implementa todo esto.

## Resumen corto
Un **perceptrón multicapa (MLP)** es una red de neuronas organizada en capas (entrada → ocultas → salida). Cada neurona calcula `z = θᵀ·[1, entradas]` (incluye bias) y aplica una **activación no lineal** (sigmoid). La red predice con **feedforward** y aprende ajustando pesos con **backpropagation** (descenso de gradiente sobre la entropía cruzada).

## Resumen completo
### Feedforward (predicción)
Para cada capa: añadir columna de 1s (bias) → `z = a_prev · θᵀ` → `a = sigmoid(z)`. La última capa da las activaciones de salida; la clase predicha = `argmax`.

### Función de coste
Entropía cruzada: `J = -(1/m)·Σ[y·log(ŷ) + (1−y)·log(1−ŷ)] + (λ/2m)·Σθ²`.
La regularización L2 NO aplica al bias (columna 0). Se hace clip de ŷ a `[ε, 1−ε]` para evitar `log(0)=NaN`.

### Backpropagation (aprendizaje)
1. Error de la capa de salida: `δ_last = a_last − y` (¡por eso `y` debe estar en **one-hot**!).
2. Propagar hacia atrás: `δ = (δ_next · θ[:,1:]) * sigmoidPrime(a)`, con `sigmoidPrime(a)=a·(1−a)`.
3. Gradiente de cada θ: `grad = (1/m)·(δᵀ · [1, a_prev]) + regL2`.
4. Actualizar: `θ := θ − α·grad`. Repetir `numIte` iteraciones.

### Arquitectura
- **Entrada** = nº de features (tras OHE). **Salida** = nº de clases (multiclase, softmax/sigmoid) o 1 (binario).
- El examen exige **probar con ≥1 (a veces ≥3) capas ocultas**, aunque el modelo final tenga otra config.

## Conceptos clave
- Neurona, peso (θ), bias, capa oculta, activación (sigmoid/relu), softmax.
- One-hot del target, feedforward, backprop, learning rate α, iteraciones, regularización λ, ε de inicialización.
- **No linealidad** viene de la activación; sin ella la red es lineal.

## Preguntas frecuentes
- *"¿Neuronas de entrada/salida?"* → ver `00_TEORIA_EXPRESS` Familia 4 (OHE).
- *"MLP no clasifica bien imágenes"* → extractor de características (CNN/autoencoder) antes (Familia 7).
- *"¿Cómo lo entrenarías?"* → OHE de la entrada categórica y de la salida; backprop minimizando entropía cruzada.

## Errores habituales (TUS errores en la ordinaria)
- ❌ No poner `y` en **one-hot** para tu MLP (el coste y `δ=a−y` lo exigen). Causa accuracy de azar.
- ❌ Llamar a las funciones helper rotas (`feedforward(X)[2]`). La API real: salida = `feedforward(X)[0][-1]`.
- ❌ No escalar las features → sigmoid se satura y no aprende.
- ❌ Pocas iteraciones / α mal → coste no baja.

## Ejercicios recomendados
- Entrenar tu MLP sobre `shopping_data.csv` o el dataset del examen. Ver `03_PRACTICA/03_usar_mi_MLP.md`.
- Reproducir Practica4 (dígitos MNIST con `ex3data1.mat`).

## Vídeos de apoyo (de `Teoria/videos_apoyo.md`)
- 3Blue1Brown "But what is a neural network?" y backpropagation (muy recomendados para intuición).
