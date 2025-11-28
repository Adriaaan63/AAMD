import numpy as np
import Utils
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Cargar datos preprocesados
X = np.load("X_train.npy")
y = np.load("y_train.npy")

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Conifgurar el modelo MLP 
clf = MLPClassifier(
    hidden_layer_sizes=(50, 25),
    activation='relu',
    solver='adam',
    alpha=0.001,
    learning_rate_init=0.001,
    max_iter=1000,
    random_state=42,
    verbose=True        # para poder ver el progreso
)

clf.fit(X_train, y_train)

# Evaluar el modelo
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n>> Precisión Final: {acc*100:.2f}%")
print(classification_report(y_test, y_pred))

# 5. Exportar usando tu Utils.py
print("Exportando archivos para Unity...")
Utils.ExportAllformatsMLPSKlearn(
    clf, 
    X_train, 
    "mlp.pkl",          # Pickle para Python
    "mlp.onnx",         # ONNX para Unity (si usáis Barracuda)
    "mlp.json",         # JSON para Unity (formato simple)
    "mlp_custom.txt"    # Custom format
)
#Archivos a llevar a Unity: 'StandardScaler.txt' y los cuatro archivos generados aquí.