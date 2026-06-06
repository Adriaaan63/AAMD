# 03.1 · Limpieza, imputación, OHE, escalado y partición

> Ej1 del examen (0.5-1 pt) + base de todo. Justifica cada decisión en una celda markdown (da puntos).

## Paso 0 — Cargar y explorar
```python
import pandas as pd
import numpy as np

df = pd.read_csv("Customer.csv")   # ruta RELATIVA, csv en el mismo zip
df.head()
df.info()                 # tipos y nulos
df.describe(include='all')
df.isna().sum()           # nº de nulos por columna
df['Segmentation'].value_counts()   # ver balance de clases (target)
```

## Paso 1 — Eliminar columnas inútiles
Quitar identificadores y columnas sin valor predictivo.
```python
df = df.drop(columns=['ID'])        # IDs no aportan; axis=1 = columna
# quita también columnas constantes o irrelevantes que detectes
```
> 📝 Markdown: "Elimino ID porque es un identificador único sin poder predictivo."

## Paso 2 — Imputar nulos (el examen pide imputar AL MENOS un atributo y justificar)
- **Numérica** → media o mediana (mediana si hay outliers).
- **Categórica** → moda (valor más frecuente).
```python
# Numérica con mediana
df['Work_Experience'] = df['Work_Experience'].fillna(df['Work_Experience'].median())
# Categórica con moda
df['Profession'] = df['Profession'].fillna(df['Profession'].mode()[0])
# Alternativa: eliminar filas con nulos si son pocas
# df = df.dropna()
```
> 📝 Markdown: "Imputo Work_Experience con la mediana porque es numérica y la mediana es robusta a valores extremos. Imputo Profession con la moda por ser categórica."

Con sklearn (alternativa válida):
```python
from sklearn.impute import SimpleImputer
num_imp = SimpleImputer(strategy='median')
df[num_cols] = num_imp.fit_transform(df[num_cols])
```

## Paso 3 — Separar X e y
```python
target = 'Segmentation'
X = df.drop(columns=[target])
y = df[target]
```

## Paso 4 — Codificar categóricas (One-Hot Encoding) en X
```python
X = pd.get_dummies(X, drop_first=False)   # cada categoría → columna 0/1
X = X.astype(float)
```
> 📝 Markdown: "Aplico One-Hot Encoding a las variables categóricas (Gender, Profession…) para que sean numéricas y aplicables a cualquier modelo."

## Paso 5 — Codificar el target y
```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_enc = le.fit_transform(y)        # 'A','B','C','D' -> 0,1,2,3  (para sklearn)
n_clases = len(le.classes_)
```
Para **tu MLP** necesitarás además el **one-hot** del target (ver `03_usar_mi_MLP.md`):
```python
y_onehot = pd.get_dummies(y).to_numpy().astype(float)   # (m, n_clases)
```

## Paso 6 — Partición train/test (¡respeta random_state y test_size del enunciado!)
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X.to_numpy(), y_enc, test_size=0.20, random_state=13, stratify=y_enc)
```
> ⚠️ El enunciado FIJA `random_state` y `test_size` (p.ej. rs=13/20% en 2026, rs=0/25% en 2025, rs=42 en árbol 2024). Úsalos exactos o no llegarás al accuracy pedido.

## Paso 7 — Escalar (imprescindible para MLP y KNN)
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)   # fit SOLO con train
X_test  = scaler.transform(X_test)        # test usa la misma escala
```
> ⚠️ `fit_transform` solo en train; en test solo `transform` (evita data leakage).

## Resultado
Tienes: `X_train, X_test` (numéricos, escalados), `y_train, y_test` (enteros para sklearn) y `y_onehot` (para tu MLP). Listo para entrenar cualquier modelo.

## Errores que te costaron puntos antes
- ❌ No imputar / no justificar en markdown.
- ❌ Olvidar OHE → el modelo recibe strings y peta.
- ❌ Escalar antes de partir (leakage) o no escalar (MLP/KNN fallan).
- ❌ No usar el `random_state` exacto del enunciado.
