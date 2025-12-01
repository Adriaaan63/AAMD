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

# Cargar
df = pd.read_csv(FILE_CSV)

# Limpieza
columns_to_erase = ['time',
                    'CAN_FIRE', 
                    'COMMAND_CENTER_X',
                    'COMMAND_CENTER_Y']

for col in columns_to_erase: 
    if col in df.columns:
        df = df.drop(columns=[col])
        print(f">> Columna '{col}' eliminada.")
        
df = df[df['action'].isin([0, 1, 2, 3, 4])] # Solo movimientos

feature_names = list(df.drop(columns=['action']).columns)

# Separar X e y
X = df.drop(columns=['action']).values
y = df['action'].values

# Normalizar con StatndarScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Guardar Scaler para Unity
mean = scaler.mean_
std = np.sqrt(scaler.var_)
if not os.path.exists(EXPORT_DIR):
    os.makedirs(EXPORT_DIR)
with open(os.path.join(EXPORT_DIR, "StandardScaler.txt"), "w") as f:
    f.write(",".join(map(str, mean)) + "\n")
    f.write(",".join(map(str, std)) + "\n")
print(">> StandardScaler.txt generado.")

# Visualizar grafica PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.5, s=10)
plt.colorbar(label='Acción')
plt.title('PCA: Distribución de Movimientos')
plt.savefig(os.path.join(EXPORT_DIR, 'grafica_pca.png'))
print(">> Gráfica PCA generada.")

# Guardar los datos
np.save(os.path.join(EXPORT_DIR, 'X_train.npy'), X_scaled)
np.save(os.path.join(EXPORT_DIR, 'y_train.npy'), y)
print(">> Datos .npy guardados.")

# feature_order.json: lista de nombres de columnas en el orden exacto usado para X
os.makedirs(EXPORT_DIR, exist_ok=True)
with open(os.path.join(EXPORT_DIR, 'feature_order.json'), 'w') as f:
    json.dump(feature_names, f, indent=2)
print(">> feature_order.json generado.")
