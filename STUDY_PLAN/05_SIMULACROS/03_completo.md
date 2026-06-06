# Simulacro COMPLETO (examen entero · modo real)

> Teórico 50 min + Práctico 2h. Sin ayudas (salvo tus apuntes, como en el examen real). Dataset práctico: usa `dementia_dataset.csv` o `shopping_data.csv`. Soluciones al final.

---

## PARTE TEÓRICA (2 pts · 50 min)

**P1 (0.25).** Un modelo da train 95%, validation 75%, test 73%, baseline 80%. Diagnostica el modelo, los datos y el entrenamiento.

**P2 (0.25).** Matriz de confusión (filas = predicho, columnas = real):
```
Modelo A   real T  real F        Modelo B   real T  real F
pred T       90      30          pred T       70       2
pred F       10      70          pred F       30      98
```
¿Qué puedes decir de cada modelo? ¿Cuándo preferirías cada uno?

**P3 (0.25).** Tienes un dataset de fotos de cartas de un juego (256×256 px) y un MLP no clasifica bien el tipo de carta. ¿Qué harías ANTES del MLP para mejorar? Justifica.

**P4 (0.25).** Tienes datos de jugadores SIN etiquetar y quieres descubrir perfiles de comportamiento. ¿Qué enfoque y técnicas usarías?

**P5 (0.25).** Un agente debe aprender a moverse por un nivel para llegar a una meta; no tienes datos de partidas previas. ¿Qué tipo de aprendizaje? ¿Qué problema principal y cómo lo abordas?

**P6 (0.25).** Quieres imitar a un jugador. Estado: posición (casilla) + 4 casillas adyacentes (cada una: vacío, muro, enemigo, item). Acciones: arriba, abajo, izq, der, disparar. ¿Cuántas neuronas de entrada y salida tendría tu MLP?

**P7 (0.25).** Entrenas y según el `random_state` la validación oscila entre 0.68 y 0.88. ¿Qué pasa y cómo lo solucionas?

**P8 (0.25).** ¿Cuál de estos es no lineal? (a) MLP con activación `y=2z+1`; (b) red con 3 capas convolucionales y una Linear final, sin activaciones; (c) árbol ID3; (d) MLP con sigmoid. Justifica.

---

## PARTE PRÁCTICA (8 pts · 2h)
Mismo guion que `05_SIMULACROS/02_intermedio.md` pero añadiendo: Ej extra **(1pt)** modifica tu MLP a `MLPRelu` con `function="relu"`, `out_function="softmax"` y demuestra que da accuracy similar al sigmoid. (Usa `03_PRACTICA/05` y `06`.)

---
---

## SOLUCIONES TEÓRICAS

**P1.** Train (95%) supera el baseline (80%) → el modelo **sí aprende**. Pero hay un **gap grande** train(95) vs val(75) → **overfitting / alta varianza**: no generaliza. Test≈val (73≈75) → el test es coherente con validación (no hay problema de distribución). Acción: **más datos** y/o **más regularización**, modelo algo más simple, data augmentation.

**P2.** A: precision = 90/(90+30)=0.75, recall = 90/(90+10)=0.90. B: precision = 70/(70+2)=0.97, recall = 70/(70+30)=0.70. → **A tiene más recall** (captura más positivos pero con más falsos positivos); **B tiene más precisión** (casi no se equivoca al decir positivo, pero se le escapan positivos). Preferir **A** cuando NO detectar un positivo es caro (diagnóstico médico, detección de fraude). Preferir **B** cuando un falso positivo es caro (bloquear usuario legítimo, alarma costosa).

**P3.** Usar un **extractor de características** antes del MLP: una **CNN** o un **autoencoder** que reduzca la imagen a un vector de características relevantes; también ayuda escalar, pasar a escala de grises, recortar y **data augmentation**. El MLP clasifica sobre esas características, no sobre los 256×256 píxeles crudos. Justificación: el MLP plano no capta estructura espacial ni escala bien con tantos píxeles.

**P4.** **Aprendizaje no supervisado**: no hay etiquetas. **Clustering** (K-Means, eligiendo k con el método del codo; o jerárquico con dendrograma) para descubrir perfiles, y **PCA** para reducir dimensión y visualizar los grupos. Escalar antes.

**P5.** **Aprendizaje por refuerzo** (no hay datos previos; aprende por prueba/error con recompensas). Problema principal: **recompensa escasa** si solo se premia llegar a la meta en un espacio grande → diseñar **recompensas intermedias** (acercarse a la meta) y, si el espacio de estados es enorme, **discretizar/comprimir** o usar **Deep RL**.

**P6.** Salida = **5** (5 acciones). Entrada: 4 casillas adyacentes con OHE de 4 valores = 4×4 = 16, más la posición (2 coords) = **18 neuronas de entrada**. (Si la posición se diera como casilla categórica con OHE, ajustar en consecuencia.)

**P7.** **Alta varianza** por **pocos datos** (la partición influye mucho). Solución: **validación cruzada (k-fold)** para una estimación estable, conseguir **más datos**, regularizar, y fijar `random_state` para reproducibilidad.

**P8.** **(d) MLP con sigmoid** es no lineal. (a) activación lineal `y=2z+1` ⇒ red lineal; (b) sin activaciones no lineales ⇒ lineal; (c) el profesor trata ID3 como lineal. → la no linealidad la aporta **la activación no lineal**.

## Corrección práctica
Usa la rúbrica de `06_SEGUIMIENTO/checklist_examen.md`. Apunta tus fallos en `06_SEGUIMIENTO/registro_errores.md`.

## Nota orientativa para aprobar
- Teórico: con 5-6 preguntas razonadas bien → ~1.3-1.6/2.
- Práctico: limpieza+scatter+MLP+2 modelos+análisis ejecutando → ~5-6/8.
- **Total ≈ 6.5-7.5/10. Aprobado.**
