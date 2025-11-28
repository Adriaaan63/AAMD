import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

FILE_CSV = 'TankTraining_Victorias_Filtradas.csv'
ACCION_DISPARO = 5

print("Cargando datos desde:", FILE_CSV)

df = pd.read_csv(FILE_CSV)
# Se quita la columna de tiempo si existe
if 'time' in df.columns:
    df = df.drop(columns=['time'])

print(f"Datos cargados: {df.shape}")
print(f"Muestras originales: {len(df)}")

# Como el agente siempre va a disparar , limpiamos las acciones de disparo
df = df[df['action'] != ACCION_DISPARO]

print(f"Muestras después de limpiar acciones de disparo: {len(df)}")

# QUITAR EN EL FUTURO, SOLO PARA VISUALIZAR MEJOR
# --- CHEQUEO DE SEGURIDAD (BALANCE) ---
conteo = df['action'].value_counts()
print("\nDistribución de acciones restantes:")
print(conteo)

if 0 not in conteo or conteo[0] < 100:
    print("\nADVERTENCIA: Te quedan muy pocos ejemplos de 'Quedarse Quieto' (Acción 0).")
    print("   El tanque podría moverse sin parar. Si ves que vibra mucho, recupera los disparos como 'Stop'.")
else:
    print("\nBalance correcto: Hay suficientes ejemplos de parada.")

# Seprar X e y
# X todos los datos que no son la accion
X = df.drop(columns=['action'])
# y son todas las acciones
y = df['action']

# Estandarizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Visualizacion PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 7))
# Mapa de colores para las acciones
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.6, s=10)
plt.colorbar(scatter, label='Acción (0:Stop, 1-4:Mover)')
plt.title('PCA: Distribución del Movimiento (Sin Disparos)')
plt.xlabel('Componente 1')
plt.ylabel('Componente 2')
plt.savefig('grafica_pca_limpia.png')
print(">> Gráfica PCA generada.")

# Exportamos
# Scaler para Unity
media = scaler.mean_
varianza = scaler.var_
std = np.sqrt(varianza)

with open("StandardScaler.txt", "w") as f:
    f.write("mean:" + ",".join(map(str, media)) + "\n")
    f.write("std:" + ",".join(map(str, std)) + "\n")

np.save('X_train.npy', X_scaled)
np.save('y_train.npy', y.values)

print(">> Datos exportados: X_train.npy, y_train.npy, StandardScaler.txt")


