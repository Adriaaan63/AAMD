from combine_data_sets import combine
from data_mining import mine_data
from training import train_sklearn_mlp, load_and_split_data

if __name__ == "__main__":
    # Primero combinamos raw data sets en uno solo
    # Genera un archivo TankTraining_Victorias_Filtradas.csv
    combine() 

    # Limpiamos y normalizamos datos 
    # Generamos los datos:
    # StandarScaler.txt --> exportacion a Unity
    # X_train.npy // y_train.npy  --> datos para entrenamiento
    # grafica_pca.png // feature_order.json --> visualizacion de datos
    mine_data()

    # Creamos el MLPClassifier, entrenamos y lo exportamos a unity
    X_train, X_test, y_train, y_test = load_and_split_data()
    train_sklearn_mlp(X_train, X_test, y_train, y_test)
    

