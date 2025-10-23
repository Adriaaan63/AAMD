import numpy as np
from matplotlib import pyplot
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder

"""
Displays 2D data stored in X in a nice grid.
"""
def displayData(X, example_width=None, figsize=(10, 10)):

    # Compute rows, cols
    if X.ndim == 2:
        m, n = X.shape
    elif X.ndim == 1:
        n = X.size
        m = 1
        X = X[None]  # Promote to a 2 dimensional array
    else:
        raise IndexError('Input X should be 1 or 2 dimensional.')

    example_width = example_width or int(np.round(np.sqrt(n)))
    example_height = n / example_width

    # Compute number of items to display
    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m / display_rows))

    fig, ax_array = pyplot.subplots(
        display_rows, display_cols, figsize=figsize)
    fig.subplots_adjust(wspace=0.025, hspace=0.025)

    ax_array = [ax_array] if m == 1 else ax_array.ravel()

    for i, ax in enumerate(ax_array):
        ax.imshow(X[i].reshape(example_width, example_width, order='F'),
                  cmap='Greys', extent=[0, 1, 0, 1])
        ax.axis('off')

"""
Load data from the dataset.
"""
def load_data(file):
    data = loadmat(file, squeeze_me=True)
    x = data['X']
    y = data['y']
    return x,y

"""
Load weights from the weights file 
"""
def load_weights(file):
    weights = loadmat(file)
    theta1, theta2 = weights['Theta1'], weights['Theta2']
    return theta1, theta2


"""
Implementation of the one hot encoding... You must use OneHotEncoder function of the sklern library. 
Probably need to use reshape(-1, 1) to change size of the data
"""
def one_hot_encoding(Y):
    """
    Convierte un vector de etiquetas (m,) con valores 0..K-1
    en una matriz one-hot (m, K) usando sklearn.OneHotEncoder.
    """
    Y = np.array(Y)

    # Asegurar que tenga forma (m, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    # Crear el codificador con categorías automáticas
    encoder = OneHotEncoder(sparse_output=False, categories='auto')
    Y_onehot = encoder.fit_transform(Y)

    return Y_onehot
    

"""
Implementation of the accuracy metrics function
"""
def accuracy(P, Y):
    P = np.array(P)
    Y = np.array(Y)

    # Si P son probabilidades -> pasar a índices
    if P.ndim == 2:
        P_idx = np.argmax(P, axis=1)
    else:
        P_idx = P.flatten().astype(int)

    # Si Y es one-hot -> pasar a índices
    if Y.ndim == 2 and Y.shape[1] > 1:
        Y_idx = np.argmax(Y, axis=1)
    else:
        Y_idx = Y.flatten().astype(int)

    if P_idx.shape[0] != Y_idx.shape[0]:
        raise ValueError("P y Y deben tener el mismo número de ejemplos")

    acc_frac = np.mean(P_idx == Y_idx)   # 0..1
    return acc_frac
