import numpy as np
import os
import sys
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

from models.MLP import MLP


EXPORT_DIR_DATA_MINING = "exports/data_mining"
EXPORT_DIR_OPTIMIZATION = "exports/optimization"

if not os.path.exists(EXPORT_DIR_OPTIMIZATION):
    os.makedirs(EXPORT_DIR_OPTIMIZATION)

print(">>> Cargando Datos...")
try:
    X = np.load(os.path.join(EXPORT_DIR_DATA_MINING, 'X_train.npy'))
    y = np.load(os.path.join(EXPORT_DIR_DATA_MINING, 'y_train.npy'))
except:
    print("Error: Ejecuta primero data_mining.py")
    sys.exit()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# OPTIMIZACION DE SKLEARN MLP
# ==========================================
def optimize_sklearn():
    print("\n--- OPTIMIZANDO SKLEARN MLP ---")
    
    # Definimos la "Rejilla" de opciones a probar
    param_grid = {
        'hidden_layer_sizes': [(50, 30), (100,), (128, 64), (50, 50, 50)],
        'activation': ['relu', 'tanh', 'logistic'],
        'solver': ['adam', 'lbfgs'], # lbfgs es mejor para pocos datos, adam para muchos
        'alpha': [0.0001, 0.01],
        'learning_rate_init': [0.001, 0.01]
    }

    mlp = MLPClassifier(max_iter=2000, random_state=42)
    
    # GridSearchCV prueba todas las combinaciones usando Cross-Validation (cv=3)
    grid = GridSearchCV(mlp, param_grid, cv=3, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    print(f"MEJOR SKLEARN: {grid.best_score_*100:.2f}%")
    print(f"   Params: {grid.best_params_}")
    
    return f"SKLEARN MLP | Acc: {grid.best_score_*100:.2f}% | Params: {grid.best_params_}\n"

# ==========================================
# OPTIMIZACION DE KNN
# ==========================================
def optimize_knn():
    print("\n--- OPTIMIZANDO KNN ---")
    
    param_grid = {
        'n_neighbors': [1, 3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    
    knn = KNeighborsClassifier()
    grid = GridSearchCV(knn, param_grid, cv=3, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    
    print(f"MEJOR KNN: {grid.best_score_*100:.2f}%")
    print(f"   Params: {grid.best_params_}")
    
    return f"KNN | Acc: {grid.best_score_*100:.2f}% | Params: {grid.best_params_}\n"

# ==========================================
# OPTIMIZACION DE RANDOM FOREST
# ==========================================
def optimize_rf():
    print("\n--- OPTIMIZANDO RANDOM FOREST ---")
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'criterion': ['gini', 'entropy']
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    
    print(f"MEJOR RF: {grid.best_score_*100:.2f}%")
    print(f"   Params: {grid.best_params_}")
    
    return f"RANDOM FOREST | Acc: {grid.best_score_*100:.2f}% | Params: {grid.best_params_}\n"

# ==========================================
# OPTIMIZACION DE CUSTOM MLP
# ==========================================
def optimize_custom_mlp():
    print("\n--- OPTIMIZANDO CUSTOM MLP (Manual) ---")
    
    encoder = OneHotEncoder(sparse_output=False)
    y_train_oh = encoder.fit_transform(y_train.reshape(-1, 1))
   
   # Hacemos el 'grid' manual 
    layers_options = [[50, 30], [25, 25], [60, 40, 20], [128, 64]]
    lr_options = [0.1, 1.0, 1.5]
    reg_options = [0.0001, 0.01]
    
    best_acc = 0
    best_config = ""
    
    total_combs = len(layers_options) * len(lr_options) * len(reg_options)
    curr = 0
    
    # iteramos para todas las opciones del grid
    for layers in layers_options:
        for lr in lr_options:
            for reg in reg_options:
                curr += 1
                print(f"Prueba {curr}/{total_combs}: L={layers}, LR={lr}, Reg={reg}", end="\r")
                
                mlp = MLP(X_train.shape[1], layers, y_train_oh.shape[1], seed=42)
                # Pocas iteraciones para que sea mas rapido
                mlp.backpropagation(X_train, y_train_oh, alpha=lr, lambda_=reg, numIte=2000, verbose=0)
                
                As, _ = mlp.feedforward(X_test)
                pred = mlp.predict(As[-1])
                
                acc = np.mean(pred == y_test)
                
                if acc > best_acc:
                    best_acc = acc
                    best_config = f"Layers={layers}, LR={lr}, Reg={reg}"
                    
    print(f"\nMEJOR CUSTOM MLP: {best_acc*100:.2f}%")
    print(f"   Config: {best_config}")
    
    return f"CUSTOM MLP | Acc: {best_acc*100:.2f}% | {best_config}\n"

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    results = []
    
    # Ejecutamos las optimizaciones
    results.append(optimize_sklearn())
    results.append(optimize_knn())
    results.append(optimize_rf())
    results.append(optimize_custom_mlp())
    
    output_file = os.path.join(EXPORT_DIR_OPTIMIZATION, "mejores_resultados.txt")
    with open(output_file, "w") as f:
        f.writelines(results)
        
    print(f"\n\n>>> RESULTADOS GUARDADOS EN: {output_file}")
    print("Copia estos valores en tu script 'training_main.py' para el entrenamiento final.")