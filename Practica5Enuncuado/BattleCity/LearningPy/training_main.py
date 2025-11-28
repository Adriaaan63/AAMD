import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder

import Utils
from models.MLP import MLP

EXPORT_DIR_TRAINING = "exports/training_main"
EXPORT_DIR_DATA_MINING = "exports/data_mining"
EXPORT_DIR_KNN = "exports/knn_model"

# --- CONFIGURACION PARA COMPARACION MLP (CUSTOM - SK)
HIDDEN_LAYERS_COMPARE = [50, 30]
ITERATIONS_COMPARE = 30000
LEARNING_RATE_COMPARE = 0.1
REGULARIZATION_COMPARE = 0.0001
SEED = 42
EPSILOM_INIT = 0.12
TEST_SIZE = 0.2

# --- CONFIGURACION PARA MODELO POTENTE (SKVARIANT) ---
HIDDEN_LAYERS_VARIANT = [128, 64] 
ITERATIONS_VARIANT = 5000

# --- PLOTTING ---
def save_confusion_matrix(y_true, y_pred, title, folder, filename):
    if not os.path.exists(folder): os.makedirs(folder)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    plt.figure(figsize=(8, 8))
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(title)
    path = os.path.join(folder, filename)
    plt.savefig(path)
    plt.close()
    print(f">> Matriz de Confusión guardada en: {path}")

def save_loss_curve(costs, title, folder, filename):
    if not os.path.exists(folder): os.makedirs(folder)
    plt.figure(figsize=(10, 6))
    plt.plot(costs, color='red', linewidth=2)
    plt.xlabel('Iteraciones')
    plt.ylabel('Coste')
    plt.title(title)
    plt.grid(True)
    path = os.path.join(folder, filename)
    plt.savefig(path)
    plt.close()

# --- MODOS DE ENTRENAMIENTO ---

def train_sklearn_compare(X_train, X_test, y_train, y_test, do_export=False):
    print(f"\n>>> SKLEARN (Modo Comparación: SGD + Logistic + LR={LEARNING_RATE_COMPARE})...")
    
    
    clf = MLPClassifier(
        hidden_layer_sizes=tuple(HIDDEN_LAYERS_COMPARE),
        activation='logistic',       
        solver='sgd',                
        learning_rate='constant',   
        learning_rate_init=LEARNING_RATE_COMPARE,
        alpha=REGULARIZATION_COMPARE,            
        max_iter=ITERATIONS_COMPARE,
        batch_size=X_train.shape[0], # Full-Batch como nuestra MLP
        random_state=SEED,
        tol=1e-6,                    
        verbose=True
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy Sklearn Comparación: {acc*100:.2f}%")
    
    if hasattr(clf, 'loss_curve_'):
        save_loss_curve(clf.loss_curve_, "Curva Sklearn (SGD)", EXPORT_DIR_TRAINING, "loss_sklearn_compare.png")
    
    save_confusion_matrix(y_test, y_pred, "Matriz Confusión Sklearn (Compare)", EXPORT_DIR_TRAINING, "conf_matrix_sklearn.png")
    
    # 2. Exportación Condicional
    if do_export:
        print(">> Exportando modelo SKLEARN para Unity...")
        Utils.ExportAllformatsMLPSKlearn(
            clf, X_train,
            os.path.join(EXPORT_DIR_TRAINING, 'mlp.pkl'),
            os.path.join(EXPORT_DIR_TRAINING, 'mlp.onnx'),
            os.path.join(EXPORT_DIR_TRAINING, 'mlp.json'),
            os.path.join(EXPORT_DIR_TRAINING, 'mlp.custom')
        )

    return acc

def train_custom_compare(X_train, X_test, y_train, y_test):
    print(f"\n>>> CUSTOM MLP (Modo Comparación: SGD + Logistic + LR={LEARNING_RATE_COMPARE})...")
    
    encoder = OneHotEncoder(sparse_output=False)
    y_train_oh = encoder.fit_transform(y_train.reshape(-1, 1))
    
    mlp = MLP(X_train.shape[1], HIDDEN_LAYERS_COMPARE, y_train_oh.shape[1], seed=SEED, epsilom=EPSILOM_INIT)
    print(f"Arquitectura: {mlp.layer_sizes}")

    costs = mlp.backpropagation(
        X_train, y_train_oh, 
        alpha=LEARNING_RATE_COMPARE, 
        lambda_=REGULARIZATION_COMPARE, 
        numIte=ITERATIONS_COMPARE, 
        verbose=2000
    )

    As, _ = mlp.feedforward(X_test)
    y_pred = mlp.predict(As[-1])
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy Custom Comparación: {acc*100:.2f}%")
    
    # Gráficas
    save_loss_curve(costs, "Curva Custom MLP", EXPORT_DIR_TRAINING, "loss_custom_compare.png")
    save_confusion_matrix(y_test, y_pred, "Matriz Confusión Custom", EXPORT_DIR_TRAINING, "conf_matrix_custom.png")

    return acc

def train_sklearn_variant(X_train, X_test, y_train, y_test):
    print("\n>>> SKLEARN VARIANT (LBFGS + ReLu)...")
    clf = MLPClassifier(
        hidden_layer_sizes=tuple(HIDDEN_LAYERS_VARIANT),
        activation='relu', solver='lbfgs', max_iter=ITERATIONS_VARIANT,
        alpha=0.0001, learning_rate_init=0.001, random_state=SEED
    )
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy Variant: {acc*100:.2f}%")
    
    save_confusion_matrix(y_test, y_pred, "Matriz Confusión Variant", EXPORT_DIR_TRAINING, "conf_matrix_variant.png")
    
    return acc

def train_knn(X_train, X_test, y_train, y_test):
    print("\n>>> KNN (Ejercicio 7)...")
    if not os.path.exists(EXPORT_DIR_KNN): os.makedirs(EXPORT_DIR_KNN)
    
    knn = KNeighborsClassifier(n_neighbors=5, metric='manhattan', weights='distance') 
    knn.fit(X_train, y_train)
    
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy KNN: {acc*100:.2f}%")
    
    save_confusion_matrix(y_test, y_pred, "Matriz Confusión KNN", EXPORT_DIR_KNN, "conf_matrix_knn.png")
    
    return acc

# --- MAIN ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-compare', action='store_true', help="Ejecutar Comparación (Custom vs Sklearn)")
    
    parser.add_argument('-sk', nargs='?', const='run_only', help="Ejecutar SKLearn. Uso: '-sk' o '-sk export'")
    
    parser.add_argument('-custom', action='store_true', help="Ejecutar Custom MLP")
    parser.add_argument('-skvariant', action='store_true', help="Entrenar SKLearn Potente (LBFGS)")
    parser.add_argument('-knn', action='store_true', help="Ejecutar KNN")
    
    args = parser.parse_args()
    
    if not any(vars(args).values()):
        print("Uso: python training_main.py [-compare] [-sk [export]] [-custom] [-skvariant] [-knn]")
        exit()

    try:
        X = np.load(os.path.join(EXPORT_DIR_DATA_MINING, 'X_train.npy'))
        y = np.load(os.path.join(EXPORT_DIR_DATA_MINING, 'y_train.npy'))
    except:
        print("Error cargando datos. Ejecuta data_mining.py")
        exit()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)
    
    if not os.path.exists(EXPORT_DIR_TRAINING): os.makedirs(EXPORT_DIR_TRAINING)

    
    if args.compare:
        print("--- EJERCICIO 4: COMPARACIÓN ---")
        acc_sk = train_sklearn_compare(X_train, X_test, y_train, y_test, do_export=False)
        acc_custom = train_custom_compare(X_train, X_test, y_train, y_test)
        print(f"\nDiferencia: {abs(acc_sk - acc_custom)*100:.2f}%")
        if acc_sk > 0.8 and acc_custom > 0.8:
            print("OBJETIVO CUMPLIDO: Ambos > 80% con mismos parametros.")
        else:
            print("AVISO: No se llegó al 80%.")

   
    if args.sk:
        should_export = (args.sk == 'export')
        train_sklearn_compare(X_train, X_test, y_train, y_test, do_export=should_export)

    if args.custom: 
        train_custom_compare(X_train, X_test, y_train, y_test)

    if args.skvariant:
        train_sklearn_variant(X_train, X_test, y_train, y_test)

    if args.knn:
        train_knn(X_train, X_test, y_train, y_test)