from MLP import MLP, target_gradient, costNN, MLP_backprop_predict
from utils import load_data, load_weights,one_hot_encoding, accuracy
from public_test import checkNNGradients,MLP_test_step
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score



"""
Test 1 to be executed in Main
"""
def gradientTest():
    checkNNGradients(costNN,target_gradient,0)
    checkNNGradients(costNN,target_gradient,1)


"""
Test 2 to be executed in Main
"""
def MLP_test(X_train,y_train, X_test, y_test):

    num_iterations = 2000
    seed = 0
    alpha_train = 1.0
    lambdas = [0.0, 0.5, 1.0]  # lambda_ a probar (regularización)

    print("We assume that: random_state of train_test_split  = 0 alpha=1, num_iterations = 2000, test_size=0.33, seed=0 and epislom = 0.12 ")
    for lambda_ in lambdas:
        print(f"Test para MLP propio. Calculando para lambda = {lambda_}")
        MLP_test_step(MLP_backprop_predict,alpha_train,X_train,y_train,X_test,y_test,0,num_iterations,0.92606,num_iterations/10)

        print(f"\n--- Comparación con sklearn MLPClassifier (lambda={lambda_}) ---")
        # sklearn no usa one-hot, así que convertimos
        y_train_labels = y_train.argmax(axis=1)
        y_test_labels  = y_test.argmax(axis=1)

        # OBSERVACION: SKLearn funciona actualizando en cada iteracion el learning rate, nosotros no lo actualizamos. 
        # Por eso, el valor es muy distinto a lo que da nuestra prediccion. 
        # Para que la prediccion de SKLearn funcione parecida a la nuestra sería necesario usar
        # solver = 'sgd'; poner el batch_size = X_train.shape[0] para que deje de ser estocástico;  poner el learning_rate = 'constant'
        # y desactivar el momentum (= 0)
        clf = MLPClassifier(
            hidden_layer_sizes=(25,),
            activation='logistic',       # misma función de activación
            solver='adam',               # optimizador del enunciado
            alpha=lambda_,               # regularización L2 = lambda_
            learning_rate_init=alpha_train,        # tu alpha = 1
            max_iter=num_iterations,               # igual que tu entrenamiento
            random_state=seed,
            tol=1e-9                     # para evitar que pare antes de tiempo
        )
        clf.fit(X_train, y_train_labels)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test_labels,y_pred)
        print(f"Accuracy sklearn MLPClassifier = {acc:.5f}")


def main():
    print("Main program")
    #Test 1
    print("\n== Running gradient checks (Test 1) ==")
    gradientTest()

    ## TO-DO: descoment both test and create the needed code to execute them.
    print("\n== Preparing data for Test 2 ==")
    X, y = load_data('data/ex3data1.mat')
    Y = one_hot_encoding(y)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=0)
    #Test 2
    print("\n== Running MLP_test (Test 2) ==")
    MLP_test(x_train, y_train, x_test, y_test )

    

main()