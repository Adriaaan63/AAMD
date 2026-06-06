# 03.5 · MLPRelu — relu + softmax (Ej6 — 1 pt)

> El examen pide una **copia** de tu MLP llamada `MLPRelu` con `function="sigmoid"|"relu"` (capas ocultas) y `out_function="sigmoid"|"softmax"` (salida). Pistas del enunciado: derivada de relu = `(a>0)*1`; el coste con relu usa `scipy.special.xlogy` para evitar `log(0)=NaN`.

## ✅ Implementación lista (NO la copies aquí: ya está en el módulo)
La clase `MLPRelu` ya está implementada y probada en **`../07_PLANTILLA_EXAMEN/practica5_mlp.py`**. Hereda de `MLP` y solo redefine activaciones, coste y gradientes. Úsala importándola:
```python
from practica5_mlp import MLP, MLPRelu, to_onehot
```
Esta guía explica **por qué** funciona, para que sepas defenderla y ajustarla.

## Por qué funciona el cambio
- **Softmax** en la salida + **entropía cruzada** → el error de la capa de salida sigue siendo `δ = a − y` (igual que con sigmoid). Por eso casi no cambia el backprop en la última capa.
- En las **capas ocultas** solo cambia la **derivada**: sigmoid → `a·(1−a)`; relu → `(a>0)·1`.
- `xlogy(y, p)` calcula `y·log(p)` devolviendo 0 cuando `y=0` aunque `p=0` (evita NaN).

## Claves de la implementación (lo que cambia respecto a MLP)
| Parte | sigmoid | relu / softmax |
|---|---|---|
| Activación oculta | `1/(1+e^-z)` | `max(0,z)` |
| Derivada oculta | `a·(1−a)` | `(a>0)·1` |
| Activación salida | sigmoid | `softmax(z)` (resta el máximo por estabilidad) |
| Coste | `-(y·log(ŷ)+(1−y)·log(1−ŷ))` | `-xlogy(y,ŷ)` (softmax) |
| δ salida | `a−y` | `a−y` (igual) |

## Prueba pedida: relu + softmax con resultado similar a sigmoid
```python
from practica5_mlp import MLPRelu, to_onehot
n_clases = len(np.unique(y_train)); y_train_oh = to_onehot(y_train, n_clases)

m_sig = MLPRelu(X_train.shape[1], [64,32], n_clases, function="sigmoid", out_function="sigmoid")
m_sig.backpropagation(X_train, y_train_oh, alpha=1.0, lambda_=1.0, numIte=2000)
acc1 = np.mean(m_sig.predict(m_sig.feedforward(X_test)[0][-1]) == y_test)

m_rs = MLPRelu(X_train.shape[1], [64,32], n_clases, function="relu", out_function="softmax")
m_rs.backpropagation(X_train, y_train_oh, alpha=0.3, lambda_=1.0, numIte=2000)
acc2 = np.mean(m_rs.predict(m_rs.feedforward(X_test)[0][-1]) == y_test)
print("sigmoid:", acc1, " relu+softmax:", acc2)   # deben salir parecidos
```
> ⚠️ Con relu usa `alpha` más pequeño (0.1–0.3): relu no satura y los gradientes pueden ser grandes. Si ves NaN, baja alpha.

## Variante 2025 (más simple)
Ese año solo pedían un parámetro `output="logistic"|"softmax"` en la salida. El módulo ya lo cubre con `out_function`.
