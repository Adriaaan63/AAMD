# 03.4 · Modelos de sklearn (Ej4-5 — 3 pts) 🔴

> Son puntos casi regalados: 3-5 líneas por modelo. Usan `y` ENTERO (no one-hot) y `X` escalado.

## Plantilla común de evaluación
```python
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def evaluar(nombre, modelo, X_tr, y_tr, X_te, y_te):
    modelo.fit(X_tr, y_tr)
    pred_tr = modelo.predict(X_tr)
    pred_te = modelo.predict(X_te)
    acc_tr = accuracy_score(y_tr, pred_tr)
    acc_te = accuracy_score(y_te, pred_te)
    print(f"{nombre}: train={acc_tr:.3f}  test={acc_te:.3f}")
    cm = confusion_matrix(y_te, pred_te)
    ConfusionMatrixDisplay(cm).plot(); plt.title(nombre); plt.show()
    return acc_tr, acc_te
```

## MLPClassifier (Ej4)
```python
from sklearn.neural_network import MLPClassifier
mlp_sk = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu',
                       max_iter=1000, random_state=0)
evaluar("MLPClassifier", mlp_sk, X_train, y_train, X_test, y_test)
```

## KNN (Ej5)
```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)   # prueba k=3,5,7,9
evaluar("KNN", knn, X_train, y_train, X_test, y_test)
```
> Si no llega al umbral, prueba varios k en un bucle y quédate con el mejor. KNN exige datos **escalados**.

## RandomForest (Ej5)
```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=200, random_state=42)
evaluar("RandomForest", rf, X_train, y_train, X_test, y_test)
```

## DecisionTree (aparece en 2024, random_state=42)
```python
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42, max_depth=None)
evaluar("DecisionTree", dt, X_train, y_train, X_test, y_test)
```

## Subir accuracy rápido (si no llegas al mínimo)
```python
# Búsqueda sencilla de k para KNN
mejor_k, mejor_acc = None, 0
for k in [3,5,7,9,11,15]:
    m = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    a = accuracy_score(y_test, m.predict(X_test))
    if a > mejor_acc: mejor_k, mejor_acc = k, a
print("Mejor k:", mejor_k, "acc:", mejor_acc)
```
- RandomForest: subir `n_estimators`, ajustar `max_depth`, `min_samples_leaf`.
- MLPClassifier: más `max_iter`, otra `hidden_layer_sizes`, `alpha` (regularización).

## Comparar y elegir (Ej de análisis — 0.5 pt)
```python
resultados = {
    "MLP propio": acc_mlp_propio,
    "MLPClassifier": acc_mlp_sk,
    "KNN": acc_knn,
    "RandomForest": acc_rf,
}
for k,v in sorted(resultados.items(), key=lambda x:-x[1]):
    print(f"{k}: {v:.3f}")
```
> 📝 Markdown de conclusión (plantilla):
> "El modelo con mejor accuracy de test es **\_\_\_** (X%). Comparando train vs test: si train≫test hay overfitting; si ambos bajos, underfitting. Para este problema elijo **\_\_\_** porque [mejor accuracy / mejor recall en la clase importante / más robusto]. La matriz de confusión muestra que confunde sobre todo las clases \_\_\_ y \_\_\_."

## Errores comunes
- ❌ Pasar `y` en one-hot a sklearn (quiere enteros).
- ❌ No escalar para KNN/MLPClassifier.
- ❌ Cambiar el `random_state` que fija el enunciado.
