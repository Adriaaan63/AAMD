# Simulacro BÁSICO (ejercicios sueltos para coger soltura)

> Objetivo: dominar cada paso por separado, sin presión de tiempo. Usa cualquier CSV (p.ej. `dementia_dataset.csv` o `shopping_data.csv`).

## Ejercicios
1. Carga un CSV y muestra: forma, tipos, nº de nulos por columna y el balance del target.
2. Elimina las columnas de identificador. Imputa una columna numérica (mediana) y una categórica (moda). Justifica.
3. Aplica OHE a las categóricas y LabelEncoder al target. ¿Cuántas features quedan?
4. Parte en train/test (test 25%, random_state=0, stratify) y escala con StandardScaler.
5. Pinta un scatter 2D (PCA) coloreando por clase.
6. Entrena tu MLP (one-hot del target) con `[32,16]` y reporta accuracy + matriz de confusión.
7. Entrena un KNN (k=5) y un RandomForest. Compara accuracies.
8. Escribe 3 frases analizando train vs test de tu mejor modelo.

## Solución (esquema)
1. `df.shape`, `df.info()`, `df.isna().sum()`, `df[TARGET].value_counts()`.
2. `df.drop(columns=['ID'])`; `df[c].fillna(df[c].median())`; `df[c].fillna(df[c].mode()[0])`. Justificación: mediana robusta, moda para categóricas.
3. `pd.get_dummies(X)`; `LabelEncoder().fit_transform(y)`. El nº de features = numéricas + Σ categorías.
4. `train_test_split(..., test_size=0.25, random_state=0, stratify=y)`; `StandardScaler` fit en train.
5. `PCA(2).fit_transform(X_train)` + `plt.scatter` por clase. (Ver `07_PLANTILLA_EXAMEN/`.)
6. `MLP(n_in,[32,16],n_clases)` + `backpropagation(X,y_oh,1.0,1.0,3000)`; `predict(feedforward(X_test)[0][-1])`; `confusion_matrix`.
7. `KNeighborsClassifier(5)`, `RandomForestClassifier(200, random_state=42)` + `accuracy_score`.
8. Diagnóstico: comparar train vs test (overfit si train≫test). Ver `02_RESUMENES/04`.

> Si completas esto sin mirar las guías, estás listo para el simulacro intermedio.
