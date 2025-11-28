import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

import Utils
from models.MLP import MLP

EXPORT_DIR_TRAINING = "exports/training_main"
EXPORT_DIR_DATA_MINING = "exports/data_mining"

# --- CONFIGUYRACION ---
HIDDEN_LAYERS = [50, 30]
ITERATIONS = 10000
LEARNING_RATE = 1.5
REGULARIZATION_LAMBDA = 0.001
SEED = 42
EPSILOM_INIT = 0.12
TEST_SIZE = 0.2

def train_sklearn_mlp(X_train, X_test, y_train, y_test):
    print("\n>>> Entrenando MLP con SKLEARN...")

    hidden_layers = tuple(HIDDEN_LAYERS)

    clf = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation='logistic',
        solver='lbfgs',
        max_iter=ITERATIONS, 
        alpha=REGULARIZATION_LAMBDA, 
        learning_rate_init=LEARNING_RATE,
        random_state=SEED,
        verbose=True,
    )

    clf.fit(X_train, y_train)

    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"Accuracy Sklearn: {acc*100:.2f}%")

    #Exportar 
    Utils.ExportAllformatsMLPSKlearn(
        clf,
        X_train,
        os.path.join(EXPORT_DIR_TRAINING, 'mlp_sklearn_model.pkl'),
        os.path.join(EXPORT_DIR_TRAINING, 'mlp_sklearn_model.onnx'),
        os.path.join(EXPORT_DIR_TRAINING, 'mlp_sklearn_model.json'),
        os.path.join(EXPORT_DIR_TRAINING, 'mlp_sklearn_model.custom')
    )
    return acc


def train_custom_mlp(X_train, X_test, y_train, y_test):
    print("\n>>> Entrenando MLP CUSTOM...")

    encoder = OneHotEncoder(sparse_output=False)
    y_train_oh = encoder.fit_transform(y_train.reshape(-1, 1))

    input_size = X_train.shape[1]
    output_size = y_train_oh.shape[1]

    mlp = MLP(input_size, HIDDEN_LAYERS, output_size, seed=SEED, epsilom=EPSILOM_INIT)
    print(f"Arquitectura creada: {mlp.layer_sizes}")

    costs = mlp.backpropagation(
        X_train,
        y_train_oh,
        alpha=LEARNING_RATE,
        lambda_=REGULARIZATION_LAMBDA,
        numIte=ITERATIONS,
        verbose=100
    )

    As, _ = mlp.feedforward(X_test)
    y_pred = mlp.predict(As[-1])

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy Custom MLP: {acc*100:.2f}%")

    #Grafica coste
    plt.figure()
    plt.plot(costs)
    plt.xlabel('Iteraciones')
    plt.ylabel('Coste')
    plt.title('Coste durante el entrenamiento custom')
    plot_path = os.path.join(EXPORT_DIR_TRAINING, 'coste_custom_mlp.png')
    plt.savefig(plot_path)
    print(">> Gráfica de coste generada.")

    return acc

if __name__ == "__main__":
    # Gestion de argumentos
    parser = argparse.ArgumentParser(description="Entrenamiento para IA de BattleCity")
    parser.add_argument('-sk', action='store_true', help="Ejecutar modelo MLP de Sklearn")
    parser.add_argument('-custom', action='store_true', help="Ejecutar modelo propio MLP de Sklearn")
    args = parser.parse_args()

    if not args.sk and not args.custom:
        print("ERROR: Debe especificar al menos un modelo a ejecutar: -sk y/o -custom")
        exit()

    run_sk = args.sk
    run_custom = args.custom

    try:
        X = np.load(os.path.join(EXPORT_DIR_DATA_MINING, 'X_train.npy'))
        y = np.load(os.path.join(EXPORT_DIR_DATA_MINING, 'y_train.npy'))
    except:
        print("Faltan datos. Ejecuta data_mining.py primero.")
        exit()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED)
    
    if not os.path.exists(EXPORT_DIR_TRAINING):
        os.makedirs(EXPORT_DIR_TRAINING)

    acc_sklearn = None
    acc_custom = None

    if run_sk:
        acc_sklearn = train_sklearn_mlp(X_train, X_test, y_train, y_test)
    if run_custom: 
        acc_custom = train_custom_mlp(X_train, X_test, y_train, y_test)

    # --- RESUMEN FINAL ---
    print("\n=== RESUMEN DE EJECUCION ===")
    if acc_sklearn is not None:
        print(f"Sklearn (LBFGS): {acc_sklearn*100:.2f}%")
    
    if acc_custom is not None:
        print(f"Custom         : {acc_custom*100:.2f}%")

