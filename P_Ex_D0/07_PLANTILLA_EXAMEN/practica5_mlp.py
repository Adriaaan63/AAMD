"""
practica5_mlp.py
Librería de Perceptrón Multicapa basada en la Práctica 4/5 (versión corregida).
- MLP: perceptrón multicapa generalizado (sigmoid + entropía cruzada + L2).
- MLPRelu: variante con activación relu/sigmoid en ocultas y softmax/sigmoid en salida.

Uso en el notebook de examen:
    from practica5_mlp import MLP, MLPRelu, to_onehot

NOTA examen: esta librería se basa en el código de la práctica 5 (mismo algoritmo:
matrices theta con bias, feedforward, backpropagation, coste con regularización L2).
"""
import numpy as np
from scipy.special import xlogy


# ---------- activaciones (reutilizables a nivel de módulo) ----------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def to_onehot(y_int, n_clases):
    """Convierte etiquetas enteras (m,) en one-hot (m, n_clases)."""
    oh = np.zeros((y_int.shape[0], n_clases))
    oh[np.arange(y_int.shape[0]), y_int] = 1
    return oh


class MLP:
    """Perceptrón multicapa generalizado (sigmoid). hidden_list = lista de neuronas por capa oculta."""

    def __init__(self, input_size, hidden_list, output_size, seed=0, epsilom=0.12):
        np.random.seed(seed)
        self.layer_sizes = [input_size] + list(hidden_list) + [output_size]
        self.thetas = [
            np.random.uniform(-epsilom, epsilom, (self.layer_sizes[i + 1], self.layer_sizes[i] + 1))
            for i in range(len(self.layer_sizes) - 1)
        ]

    def feedforward(self, x):
        m = x.shape[0]
        a = x
        As, Zs = [a], []
        for th in self.thetas:
            a = np.hstack((np.ones((m, 1)), a))
            z = a @ th.T
            Zs.append(z)
            a = sigmoid(z)
            As.append(a)
        return As, Zs

    def compute_cost(self, yhat, y, lambda_):
        m = y.shape[0]
        yhat = np.clip(yhat, 1e-12, 1 - 1e-12)
        J = -np.sum(y * np.log(yhat) + (1 - y) * np.log(1 - yhat)) / m
        J += (lambda_ / (2 * m)) * sum(np.sum(t[:, 1:] ** 2) for t in self.thetas)
        return J

    def predict(self, output_activations):
        return np.argmax(output_activations, axis=1)

    def compute_gradients(self, x, y, lambda_):
        m = x.shape[0]
        As, _ = self.feedforward(x)
        J = self.compute_cost(As[-1], y, lambda_)
        deltas = [As[-1] - y]
        for i in range(len(self.thetas) - 1, 0, -1):
            d = deltas[0].dot(self.thetas[i][:, 1:]) * (As[i] * (1 - As[i]))
            deltas.insert(0, d)
        grads = []
        for i in range(len(self.thetas)):
            ab = np.hstack((np.ones((m, 1)), As[i]))
            g = (deltas[i].T @ ab) / m
            reg = np.zeros_like(self.thetas[i])
            reg[:, 1:] = (lambda_ / m) * self.thetas[i][:, 1:]
            grads.append(g + reg)
        return J, grads

    def backpropagation(self, x, y, alpha, lambda_, numIte, verbose=0):
        Jh = []
        for it in range(numIte):
            J, grads = self.compute_gradients(x, y, lambda_)
            for j in range(len(self.thetas)):
                self.thetas[j] -= alpha * grads[j]
            Jh.append(J)
            if verbose and (it % verbose == 0 or it == numIte - 1):
                print(f"  it {it + 1}: J={J:.4f}")
        return Jh


class MLPRelu(MLP):
    """
    Variante del MLP con activaciones configurables (Ej6 del examen).
        function     = "sigmoid" | "relu"      -> capas ocultas
        out_function = "sigmoid" | "softmax"   -> capa de salida
    Derivada relu = (a > 0) * 1. Coste con xlogy para evitar log(0)=NaN.
    """

    def __init__(self, input_size, hidden_list, output_size, seed=0, epislom=0.12,
                 function="sigmoid", out_function="sigmoid"):
        super().__init__(input_size, hidden_list, output_size, seed=seed, epsilom=epislom)
        self.function = function
        self.out_function = out_function

    def _hid(self, z):
        return relu(z) if self.function == "relu" else sigmoid(z)

    def _out(self, z):
        return softmax(z) if self.out_function == "softmax" else sigmoid(z)

    def _hidp(self, a):
        return (a > 0) * 1.0 if self.function == "relu" else a * (1 - a)

    def feedforward(self, x):
        m = x.shape[0]
        a = x
        As, Zs = [a], []
        L = len(self.thetas)
        for i, th in enumerate(self.thetas):
            a = np.hstack((np.ones((m, 1)), a))
            z = a @ th.T
            Zs.append(z)
            a = self._out(z) if i == L - 1 else self._hid(z)
            As.append(a)
        return As, Zs

    def compute_cost(self, yhat, y, lambda_):
        m = y.shape[0]
        yhat = np.clip(yhat, 1e-12, 1 - 1e-12)
        if self.out_function == "softmax":
            term = -xlogy(y, yhat)
        else:
            term = -(xlogy(y, yhat) + xlogy(1 - y, 1 - yhat))
        J = np.sum(term) / m
        J += (lambda_ / (2 * m)) * sum(np.sum(t[:, 1:] ** 2) for t in self.thetas)
        return J

    def compute_gradients(self, x, y, lambda_):
        m = x.shape[0]
        As, _ = self.feedforward(x)
        J = self.compute_cost(As[-1], y, lambda_)
        deltas = [As[-1] - y]                       # vale para softmax+CE y sigmoid+CE
        for i in range(len(self.thetas) - 1, 0, -1):
            deltas.insert(0, deltas[0].dot(self.thetas[i][:, 1:]) * self._hidp(As[i]))
        grads = []
        for i in range(len(self.thetas)):
            ab = np.hstack((np.ones((m, 1)), As[i]))
            g = (deltas[i].T @ ab) / m
            reg = np.zeros_like(self.thetas[i])
            reg[:, 1:] = (lambda_ / m) * self.thetas[i][:, 1:]
            grads.append(g + reg)
        return J, grads
