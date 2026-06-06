# 03.7 · Errores comunes (checklist anti-suspenso)

## Los que te costaron la ordinaria (1,25/8)
- [ ] **Target sin one-hot** para tu MLP → entrena mal, accuracy de azar. (`to_onehot`)
- [ ] **Llamar a funciones helper rotas** de tu `mlp.py` (`feedforward(X)[2]`, `target_gradient`, `costNN`) → `IndexError`/`ValueError`, notebook no ejecuta. Usa la **clase directamente**.
- [ ] **No escalar** X → sigmoid se satura, KNN falla.
- [ ] Pipeline copiado del proyecto Unity (tanques/ONNX) en vez de uno tabular limpio.

## De datos
- [ ] Dejar columnas string sin OHE → el modelo peta.
- [ ] No imputar nulos (o no justificarlo en markdown) → pierdes el punto del Ej1.
- [ ] Escalar antes de partir train/test → **data leakage**. Orden correcto: split → `fit_transform(train)` → `transform(test)`.
- [ ] No usar el `random_state`/`test_size` EXACTOS del enunciado → no reproduces el accuracy mínimo pedido.

## De modelos
- [ ] Pasar `y` en one-hot a sklearn (quiere enteros) o entero a tu MLP (quiere one-hot).
- [ ] Salida binaria (1 neurona) y usar `argmax` → siempre 0. Usar umbral 0.5.
- [ ] Olvidar la **prueba con ≥3 (o ≥1) capas ocultas** que pide el Ej3.
- [ ] No decir en markdown **cuál es el modelo final**.

## De entrega
- [ ] Rutas **absolutas** (`C:\Users\...`) → no encuentra el CSV en otra máquina (penalización −1).
- [ ] No incluir el CSV / tu `MLP.py` en el zip.
- [ ] No poner tu **nombre** en la primera celda y en el nombre del fichero.
- [ ] No hacer **Restart & Run All** al final → entregas algo que no ejecuta limpio.

## Regla de supervivencia
> Si algo no llega al accuracy mínimo o da error, **comenta esa parte y sigue**. Un notebook que ejecuta y hace limpieza + scatter + 2-3 modelos ya suma muchos puntos. Lo que NO ejecuta, vale 0.
