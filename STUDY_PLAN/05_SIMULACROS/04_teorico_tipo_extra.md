# Batería extra de preguntas teóricas (con respuestas)

> Practica respondiendo en voz alta o por escrito en <3 min cada una. Cubre las familias del `02_RESUMENES/00_TEORIA_EXPRESS.md`.

### 1. Train 88%, Val 86%, baseline 80%. ¿Diagnóstico?
El modelo supera el baseline y train≈val → **generaliza bien**. No hay overfitting apreciable. Se podría intentar mejorar un poco con más capacidad, pero el modelo es válido.

### 2. Train 60%, Val 58%, baseline 80%. ¿Diagnóstico y acción?
Ni train llega al baseline → **underfitting / alto sesgo**. Acción: modelo **más grande** (más capas/neuronas), más/mejores features, menos regularización, entrenar más, o cambiar de modelo.

### 3. ¿Por qué escalar antes de KNN y MLP?
KNN usa distancias: una feature con rango grande dominaría. El MLP con sigmoid se **satura** con valores grandes. Escalar (StandardScaler) pone todo en escalas comparables y acelera la convergencia.

### 4. Diferencia entre DecisionTree y RandomForest.
Un árbol único aprende reglas pero **sobreajusta** fácil (alta varianza). RandomForest es un **ensemble** de muchos árboles con aleatoriedad (bagging) que **votan**; reduce la varianza y suele ser más preciso y robusto.

### 5. ¿Para qué sirve un autoencoder y cómo se entrena?
Comprime la entrada a un **espacio latente** (encoder) y la reconstruye (decoder). Se entrena para que **salida ≈ entrada** (no supervisado). Usos: extracción de características, compresión (sprites), denoising, base de generación.

### 6. Tienes solo 150 imágenes para entrenar. ¿Estrategias?
**Data augmentation** (rotaciones, ruido, recortes), **validación cruzada**, **regularización/dropout**, **transfer learning** con un extractor preentrenado, recolectar más datos.

### 7. ¿Qué modelo para predecir el precio futuro de un producto?
Es un target **continuo** → **regresión** (lineal/otros). Se pueden combinar modelos si hay señales heterogéneas.

### 8. Matriz binaria: pred T/real T=70, FP=0, FN=30, TN=100. ¿Buen modelo?
Precision=70/70=1.0 (perfecta, 0 falsos positivos), recall=70/100=0.70 (se le escapan 30 positivos). Excelente si lo caro son los falsos positivos; insuficiente si necesitamos capturar todos los positivos (subir recall).

### 9. ¿Cuántas salidas si clasifico en 4 clases con softmax? ¿Y si es binario?
4 clases → **4 neuronas** de salida con softmax. Binario → **1 neurona** sigmoid (umbral 0.5) o 2 con softmax.

### 10. ¿Qué red para reconocer dígitos manuscritos (imágenes)?
**CNN** (capta patrones espaciales). Un MLP plano funciona en MNIST pequeño (28×28) pero la CNN es la respuesta canónica para imágenes; o extractor de características + MLP.

### 11. ¿Qué red para GENERAR un dígito que teclea el usuario?
Modelo **generativo**: autoencoder/decoder o **GAN** condicionada a la clase. Se entrena a reconstruir/generar imágenes de cada dígito; al introducir la clase, genera la imagen.

### 12. random_state cambia mucho el resultado. ¿Causa?
**Alta varianza** por pocos datos / modelo inestable. Solución: validación cruzada, más datos, regularización.
