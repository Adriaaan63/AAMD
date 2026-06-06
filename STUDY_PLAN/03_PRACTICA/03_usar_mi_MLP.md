# 03.3 · Usar TU Perceptrón Multicapa (Ej3 — 2 pts) 🔴

> Tu `MLP.py` (Práctica 5, idéntico al que entregaste) **funciona en su núcleo**. El problema fue cómo lo conectaste a los datos. Aquí está la forma CORRECTA.

## API real de tu clase MLP (lo que de verdad acepta)
```python
# Constructor: hidden_layer_size es una LISTA de neuronas por capa oculta
MLP(input_layer_size, hidden_layer_size, output_layer_size, seed=0, epsilom=0.12)
#   self.layer_sizes = [input] + hidden_layer_size + [output]

mlp.backpropagation(X, y_onehot, alpha, lambda_, numIte, verbose=0)  # entrena
As, Zs = mlp.feedforward(X)        # As[-1] = activaciones de salida
y_pred = mlp.predict(As[-1])       # argmax por fila -> clase
```

## ⚠️ NO uses estas funciones (están ROTAS en tu archivo)
- `MLP_backprop_predict(...)` → hace `feedforward(X)[2]` pero feedforward solo devuelve 2 cosas (`As, Zs`) → **IndexError**.
- `target_gradient(...)` y `costNN(...)` → esperan `J, grad1, grad2` pero `compute_gradients` devuelve `J, grads` (2 valores) → **ValueError**.
- Estas eran para los tests de la Práctica 4. **En el examen NO las llames.** Usa la clase directamente como abajo.

## Receta correcta (multiclase, p.ej. Customer A/B/C/D)
```python
import numpy as np
from models.MLP import MLP          # ajusta el import a tu estructura de carpetas

# 1) One-hot del target de ENTRENAMIENTO (imprescindible: el coste usa delta = a - y)
def to_onehot(y_int, n_clases):
    oh = np.zeros((y_int.shape[0], n_clases))
    oh[np.arange(y_int.shape[0]), y_int] = 1
    return oh

n_clases = len(np.unique(y_train))
y_train_oh = to_onehot(y_train, n_clases)

# 2) Crear y entrenar (X_train YA escalado). Prueba pedida con >=3 capas ocultas:
mlp = MLP(X_train.shape[1], [64, 32, 16], n_clases, seed=0, epsilom=0.12)
J_hist = mlp.backpropagation(X_train, y_train_oh, alpha=1.0, lambda_=1.0,
                             numIte=3000, verbose=500)

# 3) Predecir y medir
As, _ = mlp.feedforward(X_test)
y_pred = mlp.predict(As[-1])
acc = np.mean(y_pred == y_test)
print("Accuracy MLP propio:", acc)

# 4) Matriz de confusión
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot(); plt.show()
```

## Cumplir el enunciado al pie de la letra
- **"prueba con ≥3 capas ocultas"**: deja una celda con `MLP(..., [64,32,16], ...)` aunque luego te quedes con otra. Di EXPLÍCITAMENTE en markdown cuál es tu modelo final.
- **Accuracy mínimo** (50% en 2026, 84% en 2025, 65% en 2024): si no llegas, prueba más iteraciones, otra `alpha` (0.3–3), bajar `lambda_`, otra arquitectura. **Aunque no llegues, entrega el código que ejecuta.**

## Ajuste de hiperparámetros (si el accuracy es bajo)
| Síntoma | Acción |
|---|---|
| El coste (J) no baja | subir `numIte`, ajustar `alpha` (prueba 0.3, 1, 3), comprobar que escalaste X |
| Coste baja pero accuracy malo en train | red más grande `[128,64,32]`, bajar `lambda_` |
| Train bien, test mal (overfit) | subir `lambda_` (0.1→1→3), red más pequeña |
| Todo da la misma clase | seguro que **falta one-hot** del target o no escalaste |

## Caso BINARIO con 1 sola neurona de salida (Ej6 de 2024)
Si la salida es 1 neurona (output_layer_size=1), `predict` (argmax) NO sirve. Usa umbral 0.5:
```python
mlp = MLP(X_train.shape[1], [16, 8], 1, seed=0)
y_train_col = y_train.reshape(-1, 1).astype(float)   # 0/1 en columna
mlp.backpropagation(X_train, y_train_col, alpha=1.0, lambda_=0.0, numIte=3000)
As, _ = mlp.feedforward(X_test)
prob = As[-1].ravel()
y_pred = (prob >= 0.5).astype(int)
acc = np.mean(y_pred == y_test)
```

## Recordatorio de por qué fallaste antes
El núcleo estaba bien; el accuracy bajo venía de **no codificar el target en one-hot** y/o **no escalar**, y de llamar a las funciones helper rotas. Con esta receta eso queda resuelto.
