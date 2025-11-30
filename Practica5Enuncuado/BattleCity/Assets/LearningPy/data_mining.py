import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import json

# --- CONFIG ---
FILE_CSV = 'TankTraining_Victorias_Filtradas.csv'
EXPORT_DIR = "exports/data_mining"

print("=== 1. DATA MINING Y PREPROCESADO ===")

# 1. Cargar
df = pd.read_csv(FILE_CSV)
if 'time' in df.columns: df = df.drop(columns=['time'])
feature_names = list(df.drop(columns=['action']).columns)

print(f"Muestras totales: {len(df)}")

# 2. Limpieza
df = df[df['action'].isin([0, 1, 2, 3, 4])] # Solo movimientos

print(f"Muestras tras limpieza: {len(df)}")

# 3. Separar X e y
X = df.drop(columns=['action']).values
y = df['action'].values

# 4. Normalizar (StandardScaler)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Guardar Scaler para Unity
mean = scaler.mean_
std = np.sqrt(scaler.var_)
if not os.path.exists(EXPORT_DIR):
    os.makedirs(EXPORT_DIR)
with open(os.path.join(EXPORT_DIR, "StandardScaler.txt"), "w") as f:
    f.write("MEAN\n" + ",".join(map(str, mean)) + "\n")
    f.write("STD\n" + ",".join(map(str, std)) + "\n")
print(">> StandardScaler.txt generado.")

# 5. Visualizar (PCA)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.5, s=10)
plt.colorbar(label='Acción')
plt.title('PCA: Distribución de Movimientos')
plt.savefig(os.path.join(EXPORT_DIR, 'grafica_pca.png'))
print(">> Gráfica PCA generada.")

# 6. Guardar Datos Procesados
np.save(os.path.join(EXPORT_DIR, 'X_train.npy'), X_scaled)
np.save(os.path.join(EXPORT_DIR, 'y_train.npy'), y)
print(">> Datos .npy guardados.")

# 7. feature_order.json: lista de nombres de columnas en el orden exacto usado para X
os.makedirs(EXPORT_DIR, exist_ok=True)
with open(os.path.join(EXPORT_DIR, 'feature_order.json'), 'w') as f:
    json.dump(feature_names, f, indent=2)
print(">> feature_order.json generado.")
