# Simulacro INTERMEDIO (práctico cronometrado · 2h)

> Dataset: usa **`dementia_dataset.csv`** (está en el repo, examen real de 2024). Cronómetro: 2 horas. Sin mirar soluciones hasta terminar.

## Enunciado (réplica del examen 2024)
Dataset sobre demencia. Target **Group** ∈ {Demented, Nondemented, Converted}. Campos: Subject ID, MRI ID, Visit, MR Delay, M/F, Hand, Age, SES, MMSE, CDR, eTIV, nWBV, ASF.

1. **(1pt)** Limpia el dataset (quita IDs y columnas irrelevantes, trata nulos). Justifica en markdown.
2. **(1pt)** Representa gráficamente los datos limpios (scatter por clase).
3. **(2pt)** Usa tu MLP. Accuracy + matriz de confusión. Prueba con >1 capa oculta. Mínimo 65%. Di tu modelo final.
4. **(1.5pt)** Entrena un DecisionTree (`random_state=42`). Accuracy + confusión. Objetivo ~85%.
5. **(1pt)** Compara ambos modelos en markdown y elige justificadamente.
6. **(1.5pt)** Transforma Converted→Demented y crea un MLP **binario** (1 neurona de salida).

## Solución comentada

**Ej1 — limpieza:**
- Quitar `Subject ID`, `MRI ID` (identificadores), `Hand` (constante: todos diestros → sin info), `Visit`/`MR Delay` si no aportan.
- `M/F` → OHE o mapear M=0/F=1. Nulos en `SES`/`MMSE` → imputar con mediana o `dropna()` si son pocos.
```python
df = df.drop(columns=['Subject ID','MRI ID','Hand'])
df['SES'] = df['SES'].fillna(df['SES'].median())
df['MMSE'] = df['MMSE'].fillna(df['MMSE'].median())
X = pd.get_dummies(df.drop(columns=['Group']), drop_first=True).astype(float)
y = LabelEncoder().fit_transform(df['Group'])
```
> ⚠️ **CDR** está muy correlacionada con el diagnóstico (es una escala de demencia); el profesor puede considerarla "trampa". Puedes comentarlo en markdown (incluirla o no, justificándolo).

**Ej2 — scatter:** PCA 2D coloreado por clase (ver `07_PLANTILLA_EXAMEN/` celda scatter).

**Ej3 — MLP:** split + escalar → `to_onehot(y_train,3)` → `MLP(n_in,[32,16],3)` → entrenar 3000 it → confusión. Si <65%, subir iteraciones/ajustar alpha. Deja una prueba con `[64,32,16]`.

**Ej4 — DecisionTree:**
```python
dt = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
print(accuracy_score(y_test, dt.predict(X_test)))
```

**Ej5 — comparación (markdown):** "El árbol alcanza mayor accuracy y es interpretable; el MLP generaliza distinto. Para detección médica priorizo **recall** en la clase Converted/Demented (no dejar pasar enfermos). Elijo ___ por ___."

**Ej6 — binario:**
```python
y_bin = (df['Group'] != 'Nondemented').astype(int).to_numpy()   # Demented+Converted = 1
# split y escalar con y_bin
mlp_b = MLP(X_train.shape[1], [16,8], 1, seed=0)
mlp_b.backpropagation(X_train, y_train.reshape(-1,1).astype(float), 1.0, 0.0, 3000)
prob = mlp_b.feedforward(X_test)[0][-1].ravel()
y_pred = (prob >= 0.5).astype(int)
print(accuracy_score(y_test, y_pred))
```

## Autoevaluación
- [ ] El notebook ejecuta entero sin errores.
- [ ] Llegué (o me acerqué) a los umbrales.
- [ ] Justifiqué limpieza y elección de modelo en markdown.
- [ ] Lo hice en ≤2h.
