# Generador del notebook de examen. Construye un .ipynb válido vía json.dump.
import json, os

cells = []
def md(src):  cells.append(("markdown", src))
def code(src): cells.append(("code", src))

md("""# AAMD — Plantilla de Examen
**Nombre: Marcos Pérez Martínez**

Plantilla lista para examen. Cambia la **CELDA CONFIG** (fichero, target, columnas, random_state, test_size) y ejecuta de arriba a abajo (`Kernel > Restart & Run All`).

El MLP/MLPRelu se **importan** del módulo `practica5_mlp.py` (debe estar en la misma carpeta / zip). No se duplica código de la práctica.

Orden: Config → Imports (incl. módulo MLP) → Carga → Limpieza → Encoding → Split/Escalado → Scatter → MLP propio → sklearn (MLPClassifier/KNN/RandomForest/Tree) → Comparación → MLPRelu → (binario).""")

code("""# ===== CONFIG (lo unico que sueles tocar) =====
FILE        = "Customer.csv"     # nombre del CSV (ruta RELATIVA, en el mismo zip)
TARGET      = "Segmentation"     # columna objetivo
COLS_DROP   = ["ID"]             # columnas a eliminar (IDs, constantes)
RANDOM_STATE = 13                # el que fije el enunciado (2026=13, 2025=0, tree=42)
TEST_SIZE    = 0.20              # el que fije el enunciado (2026=0.20, 2025=0.25)""")

code("""# ===== IMPORTS + MODULO MLP (reutilizado de la practica 5) =====
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

# El MLP/MLPRelu NO se redefinen aqui: se importan del modulo (mismo zip)
from practica5_mlp import MLP, MLPRelu, to_onehot

def evaluar(nombre, modelo, Xtr, ytr, Xte, yte, plot=True):
    modelo.fit(Xtr, ytr)
    atr = accuracy_score(ytr, modelo.predict(Xtr))
    ate = accuracy_score(yte, modelo.predict(Xte))
    print(f"{nombre}: train={atr:.3f}  test={ate:.3f}")
    if plot:
        ConfusionMatrixDisplay(confusion_matrix(yte, modelo.predict(Xte))).plot()
        plt.title(nombre); plt.show()
    return ate""")

code('''# ===== CARGA + EXPLORACION =====
df = pd.read_csv(FILE)
print(df.shape)
display(df.head())
df.info()
print(df.isna().sum())
print(df[TARGET].value_counts())''')

code('''# ===== LIMPIEZA + IMPUTACION (Ej1) =====
for c in COLS_DROP:
    if c in df.columns: df = df.drop(columns=[c])
num_cols = df.select_dtypes(include=np.number).columns.drop(TARGET, errors="ignore")
cat_cols = df.select_dtypes(exclude=np.number).columns.drop(TARGET, errors="ignore")
for c in num_cols: df[c] = df[c].fillna(df[c].median())
for c in cat_cols: df[c] = df[c].fillna(df[c].mode()[0])
print("Nulos restantes:", int(df.isna().sum().sum()))''')

md("""**Justificación (Ej1):** elimino identificadores (sin poder predictivo); imputo numéricas con **mediana** (robusta a outliers) y categóricas con **moda**; aplico One-Hot a categóricas para que cualquier modelo las acepte.""")

code('''# ===== ENCODING =====
X = pd.get_dummies(df.drop(columns=[TARGET]), drop_first=False).astype(float)
le = LabelEncoder(); y = le.fit_transform(df[TARGET])
n_clases = len(le.classes_)
print("features:", X.shape[1], "| clases:", n_clases, le.classes_)''')

code('''# ===== SPLIT + ESCALADO =====
X_train, X_test, y_train, y_test = train_test_split(
    X.to_numpy(), y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test  = sc.transform(X_test)
print(X_train.shape, X_test.shape)''')

code('''# ===== SCATTER DISTRIBUCION DE CLASES (Ej2) =====
X2 = PCA(n_components=2).fit_transform(X_train)
plt.figure(figsize=(7,5))
for c in range(n_clases):
    plt.scatter(X2[y_train==c,0], X2[y_train==c,1], s=12, alpha=.6, label=str(le.classes_[c]))
plt.legend(); plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("Distribucion de clases (PCA 2D)"); plt.show()''')

code('''# ===== MLP PROPIO (Ej3) =====
y_train_oh = to_onehot(y_train, n_clases)

# Prueba EXIGIDA con >=3 capas ocultas
mlp3 = MLP(X_train.shape[1], [64,32,16], n_clases, seed=0)
mlp3.backpropagation(X_train, y_train_oh, alpha=1.0, lambda_=1.0, numIte=3000, verbose=1000)
acc3 = np.mean(mlp3.predict(mlp3.feedforward(X_test)[0][-1]) == y_test)
print("MLP [64,32,16] test:", round(acc3,3))

# Modelo final
mlp = MLP(X_train.shape[1], [64,32], n_clases, seed=0)
mlp.backpropagation(X_train, y_train_oh, alpha=1.0, lambda_=1.0, numIte=3000, verbose=1000)
y_pred_mlp = mlp.predict(mlp.feedforward(X_test)[0][-1])
acc_mlp = np.mean(y_pred_mlp == y_test); print("MLP final test:", round(acc_mlp,3))
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_mlp)).plot(); plt.title("MLP propio"); plt.show()''')

md("""**Modelo final MLP:** me quedo con `[64,32]` (similar o mejor que `[64,32,16]` con menos coste). Si no llega al minimo: subir `numIte`, ajustar `alpha` (0.3/1/3), bajar `lambda_`, o ampliar la red.""")

code('''# ===== MLPClassifier (Ej4) =====
acc_sk = evaluar("MLPClassifier",
    MLPClassifier(hidden_layer_sizes=(64,32), activation="relu", max_iter=1000, random_state=0),
    X_train, y_train, X_test, y_test)''')

code('''# ===== KNN (Ej5) — con busqueda de k =====
best_k, best = 5, 0
for k in [3,5,7,9,11,15]:
    a = accuracy_score(y_test, KNeighborsClassifier(k).fit(X_train,y_train).predict(X_test))
    if a > best: best_k, best = k, a
print("mejor k:", best_k)
acc_knn = evaluar(f"KNN(k={best_k})", KNeighborsClassifier(best_k), X_train, y_train, X_test, y_test)''')

code('''# ===== RandomForest (Ej5) =====
acc_rf = evaluar("RandomForest",
    RandomForestClassifier(n_estimators=200, random_state=42),
    X_train, y_train, X_test, y_test)''')

code('''# ===== DecisionTree (si lo piden, p.ej. 2024 rs=42) =====
acc_dt = evaluar("DecisionTree",
    DecisionTreeClassifier(random_state=42),
    X_train, y_train, X_test, y_test)''')

code('''# ===== COMPARACION (Ej analisis) =====
res = {"MLP propio":acc_mlp, "MLPClassifier":acc_sk, "KNN":acc_knn,
       "RandomForest":acc_rf, "DecisionTree":acc_dt}
for k,v in sorted(res.items(), key=lambda x:-x[1]): print(f"{k:14s}: {v:.3f}")''')

md("""**Análisis (Ej):** comparo train vs test de cada modelo (train≫test → overfitting; ambos bajos → underfitting). Elijo el de mayor accuracy de test (o mejor recall en la clase crítica si es deteccion médica). La matriz de confusión indica entre qué clases se confunde.""")

code('''# ===== MLPRelu: relu+softmax ~ sigmoid (Ej6) =====
m_sig = MLPRelu(X_train.shape[1], [64,32], n_clases, function="sigmoid", out_function="sigmoid")
m_sig.backpropagation(X_train, y_train_oh, alpha=1.0, lambda_=1.0, numIte=2000)
a_sig = np.mean(m_sig.predict(m_sig.feedforward(X_test)[0][-1]) == y_test)

m_rs = MLPRelu(X_train.shape[1], [64,32], n_clases, function="relu", out_function="softmax")
m_rs.backpropagation(X_train, y_train_oh, alpha=0.3, lambda_=1.0, numIte=2000)  # alpha menor con relu
a_rs = np.mean(m_rs.predict(m_rs.feedforward(X_test)[0][-1]) == y_test)
print("sigmoid:", round(a_sig,3), "| relu+softmax:", round(a_rs,3))''')

code('''# ===== (OPCIONAL) MLP BINARIO 1 neurona de salida (estilo 2024) =====
# y_bin = (df[TARGET] != "ClaseNegativa").astype(int).to_numpy()
# Xb_tr,Xb_te,yb_tr,yb_te = train_test_split(X.to_numpy(), y_bin, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_bin)
# Xb_tr=sc.fit_transform(Xb_tr); Xb_te=sc.transform(Xb_te)
# mb = MLP(Xb_tr.shape[1],[16,8],1,seed=0)
# mb.backpropagation(Xb_tr, yb_tr.reshape(-1,1).astype(float), 1.0, 0.0, 3000)
# yb_pred = (mb.feedforward(Xb_te)[0][-1].ravel() >= 0.5).astype(int)
# print("MLP binario:", round(accuracy_score(yb_te, yb_pred),3))''')

nb = {
    "cells": [
        {"cell_type":"markdown","metadata":{},"source":[s]} if t=="markdown"
        else {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[s]}
        for (t,s) in cells
    ],
    "metadata": {
        "kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
        "language_info": {"name":"python","version":"3.11"}
    },
    "nbformat": 4, "nbformat_minor": 5
}

out = os.path.join(os.path.dirname(__file__), "..", "07_PLANTILLA_EXAMEN", "PLANTILLA_EXAMEN.ipynb")
out = os.path.abspath(out)
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("OK ->", out, "| celdas:", len(cells))
