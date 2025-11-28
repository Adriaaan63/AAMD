import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder

import Utils
from models.MLP import MLP

EXPORT_DIR_TRAINING = "exports/training_main"
EXPORT_DIR_DATA_MINING = "exports/data_mining"

# --- CONFIGUYRACION ---
HIDDEN_LAYERS = [50, 30]
ITERATIONS = 10000
REGULARIZATION_LAMBDA = 0.0001
SEED = 42
EPSILOM_INIT = 0.12
TEST_SIZE = 0.2

# --- FUNCIONES DE PLOTTING --- 
def save_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    plt.figure(figsize=(8, 8))
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(title)
    path = os.path.join(EXPORT_DIR_TRAINING, filename)
    plt.savefig(path)
    plt.close()
    print(f">> Matriz de Confusión guardada en: {path}")

def save_loss_curve(costs, title, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(costs, color='red', linewidth=2)
    plt.xlabel('Iteraciones')
    plt.ylabel('Coste (Loss)')
    plt.title(title)
    plt.grid(True)
    path = os.path.join(EXPORT_DIR_TRAINING, filename)
    plt.savefig(path)
    plt.close()
    print(f">> Gráfica de Costes guardada en: {path}")

# --- FUNCIONES DE ENTRENAMIENTO ---

def train_sklearn_mlp(X_train, X_test, y_train, y_test, hidden_layers_sizes, solver, activation, max_iters, lr, reg_lambda, model_name="sklearn"):
    hidden_layers = tuple(hidden_layers_sizes)

    clf = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation=activation,
        solver=solver,     #Mejor para conjuntos de datos pequeños
        max_iter=max_iters, 
        alpha=reg_lambda, 
        learning_rate_init=lr,
        learning_rate= 'constant',
        random_state=SEED,
        verbose=True,
    )

    clf.fit(X_train, y_train)

    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"Accuracy {model_name} ({solver}): {acc*100:.2f}%")

    y_pred = clf.predict(X_test)
    save_confusion_matrix(y_test, y_pred, 
        f"Matriz Confusión {model_name}_{solver}_{activation}", 
        f"conf_matrix_{model_name}_{solver}_{activation}.png"
    )
    
    if hasattr(clf, 'loss_curve_'):
        save_loss_curve(clf.loss_curve_,
            f"Curva de Aprendizaje {model_name}_{solver}_{activation}",
            f"loss_curve_{model_name}_{solver}_{activation}.png"
        )
    else:
        print(f"AVISO: El solver '{solver}' no genera curva de pérdida en Sklearn.")

    return clf, X_train, acc

def train_sklearn_mlp_export(X_train, X_test, y_train, y_test, solver):
    print(f"\n>>> Entrenando MLP con SKLEARN (Solver: {solver}) [Modo Export]...")

    clf, X_train, acc = train_sklearn_mlp(
        X_train, X_test, y_train, y_test,
        hidden_layers_sizes=HIDDEN_LAYERS,
        solver=solver,
        activation='logistic',
        max_iters=ITERATIONS,
        lr=0.001,
        reg_lambda=REGULARIZATION_LAMBDA
    )

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
        alpha=1.5,
        lambda_=REGULARIZATION_LAMBDA,
        numIte=ITERATIONS,
        verbose=100
    )

    As, _ = mlp.feedforward(X_test)
    y_pred = mlp.predict(As[-1])

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy Custom MLP: {acc*100:.2f}%")

    save_loss_curve(costs, "Coste Custom MLP", "coste_custom_mlp.png")
    save_confusion_matrix(y_test, y_pred, "Matriz Confusión Custom MLP", "conf_matrix_custom.png")

    return acc

if __name__ == "__main__":
    # Gestion de argumentos
    parser = argparse.ArgumentParser(description="Entrenamiento IA BattleCity")
    parser.add_argument('-sk', nargs='?', const='lbfgs', default=None, 
                        help="Ejecutar Sklearn. Uso: '-sk' (usa lbfgs) o '-sk adam'")
    
    parser.add_argument('-sknex', action='store_true', help="Ejecutar Sklearn Turbo (sin export)")
    parser.add_argument('-custom', action='store_true', help="Ejecutar Custom MLP")
    args = parser.parse_args()

    if args.sk is None and not args.custom and not args.sknex:
        print("ERROR: Especifica modelo:")
        print("  python main.py -sk          (Sklearn con lbfgs)")
        print("  python main.py -sk adam     (Sklearn con adam)")
        print("  python main.py -custom      (Tu MLP)")
        print("  python main.py -sknex       (Turbo Mode)")
        exit()

    run_sk = args.sk
    run_custom = args.custom
    run_sknex = args.sknex

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
        solver = args.sk
        acc_sklearn = train_sklearn_mlp_export(X_train, X_test, y_train, y_test, solver)
    if run_custom: 
        acc_custom = train_custom_mlp(X_train, X_test, y_train, y_test)
    if run_sknex:
        print(">>> Entrenando MLP con SKLEARN (ReLu sin exportacion)...")
        train_sklearn_mlp(
            X_train, X_test, y_train, y_test,
            hidden_layers_sizes=[128, 64],
            solver='lbfgs',
            activation='relu',
            max_iters=5000,
            lr=0.01,         # IGNORADO por lbfgs, pero necesario por la función
            reg_lambda=1e-5

        )

    # --- RESUMEN FINAL ---
    print("\n=== RESUMEN DE EJECUCION ===")
    if acc_sklearn is not None:
        print(f"Sklearn (LBFGS): {acc_sklearn*100:.2f}%")
    
    if acc_custom is not None:
        print(f"Custom         : {acc_custom*100:.2f}%")

