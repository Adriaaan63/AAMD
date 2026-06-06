# T08 · Refuerzo ⚪ y T09 · IA Generativa 🟡 (prioridad baja)

> El profesor avisó: "este año no llegamos a refuerzo, no caería". Dedícale lo mínimo. Generativa puede aparecer en una pregunta corta.

## T08 · Aprendizaje por Refuerzo (lo justo)
- Un **agente** interactúa con un **entorno**, toma **acciones**, recibe **recompensas** y aprende una **política** que maximiza la recompensa acumulada. No hay dataset previo.
- **Q-Learning:** aprende una tabla Q(estado, acción) con el valor esperado de cada acción.
- Problemas: **espacio de estados enorme** → discretizar/comprimir estados o usar **Deep RL** (red en vez de tabla). Diseñar bien la **recompensa** (recompensas intermedias si la final es muy rara).
- Cuándo usarlo: agente que debe aprender **sin datos**, por prueba y error (p. ej. superar un nivel procedural sin ejemplos guardados).

## T09 · IA Generativa (lo justo)
- Modelos que **generan** datos nuevos parecidos a los de entrenamiento.
- **Autoencoder / VAE:** reconstruyen y generan desde el espacio latente.
- **GAN (Generative Adversarial Network):** un **generador** crea muestras y un **discriminador** intenta distinguir reales de falsas; compiten y el generador mejora.
- Usos en juegos: generar imágenes/sprites/texturas, contenido procedural.

## Preguntas frecuentes
- *"Generar imágenes de dígitos / sprites nuevos"* → autoencoder/GAN.
- *"Agente sin datos en espacio enorme"* → refuerzo (discretizar o Deep RL, recompensas intermedias). ⚪

## Errores habituales
- Proponer refuerzo cuando SÍ hay datos etiquetados (ahí va supervisado/MLP).
- No mencionar el problema de la recompensa escasa en espacios grandes.
