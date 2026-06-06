# T07 · Deep Learning (CNN y Autoencoders) 🟠

> Cae en TEORÍA con frecuencia (imágenes, compresión, generación). No necesitas implementarlo, sí explicarlo.

## Resumen corto
El **deep learning** usa redes con muchas capas. Para **imágenes** se usan **CNN** (capas convolucionales que detectan patrones locales). Los **autoencoders** comprimen datos a un espacio latente y los reconstruyen; sirven como extractores de características, para compresión y para generación.

## Resumen completo
### CNN (Red Neuronal Convolucional)
- Capas **convolucionales** (filtros que recorren la imagen detectando bordes, texturas…) + **pooling** (reduce tamaño) + **full-connected** finales para clasificar.
- Por qué para imágenes: explota la estructura espacial y comparte pesos → muchos menos parámetros que un MLP plano sobre píxeles.
- ⚠️ Si una CNN solo tiene capas convolucionales + Linear sin activaciones no lineales → es **lineal**.

### Autoencoder
- **Codificador** (encoder): comprime la entrada a un vector pequeño (**espacio latente**, cuello de botella).
- **Decodificador** (decoder): reconstruye la entrada desde el latente.
- Se entrena para que **salida ≈ entrada** (no necesita etiquetas → no supervisado).
- Usos: **extracción de características**, **compresión** (p. ej. sprites para ahorrar memoria 📌), **denoising**, base para **generación**.

### Generación de imágenes
- Autoencoder/**GAN** (ver T09): a partir de un código/entrada generan una imagen nueva. Para "generar el dígito que teclea el usuario" → entrenar un decodificador/generador condicionado a la clase.

## Conceptos clave
- Convolución, filtro/kernel, pooling, capas full-connected, profundidad.
- Encoder/decoder, espacio latente, reconstrucción, extractor de características.

## Preguntas frecuentes (todas en `00_TEORIA_EXPRESS`)
- *"MLP no va con imágenes 128×128"* → extractor de características (CNN/autoencoder) antes del MLP (Familia 7).
- *"Reducir memoria de sprites"* → autoencoder (Familia 3/6).
- *"Identifica esta red espejo"* → autoencoder, encoder+decoder (Familia 6).
- *"Generar imagen de un dígito"* → autoencoder/GAN como generador.

## Errores habituales
- Decir que cualquier red profunda es no lineal sin mirar las activaciones.
- Confundir "clasificar" (CNN→clase) con "generar/reconstruir" (autoencoder/GAN).
