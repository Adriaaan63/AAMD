import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder


###########################################################################
# data display
#
def displayData(X):
    num_plots = int(np.size(X, 0)**.5)
    fig, ax = plt.subplots(num_plots, num_plots, sharex=True, sharey=True)
    plt.subplots_adjust(left=0, wspace=0, hspace=0)
    img_num = 0
    for i in range(num_plots):
        for j in range(num_plots):
            # Convert column vector into 20x20 pixel matrix
            # transpose
            img = X[img_num, :].reshape(20, 20).T
            ax[i][j].imshow(img, cmap='Greys')
            ax[i][j].set_axis_off()
            img_num += 1

    return (fig, ax)


def displayImage(im):
    fig2, ax2 = plt.subplots()
    image = im.reshape(20, 20).T
    ax2.imshow(image, cmap='gray')
    return (fig2, ax2)



def load_data(file):
    data = loadmat(file, squeeze_me=True)
    x = data['X']
    y = data['y']
    return x,y

def load_weights(file):
    weights = loadmat(file)
    theta1, theta2 = weights['Theta1'], weights['Theta2']
    return theta1, theta2

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


###########################################################################
# gradient checking
#
def debugInitializeWeights(fan_in, fan_out):
    """
    Initializes the weights of a layer with fan_in incoming connections and
    fan_out outgoing connections using a fixed set of values.
    """

    W = np.sin(np.arange(1, 1 + (1+fan_in)*fan_out))/10.0
    W = W.reshape(fan_out, 1+fan_in, order='F')
    return W


def computeNumericalGradient(J, Theta1, Theta2):
    """
    Computes the gradient of J around theta using finite differences and
    yields a numerical estimate of the gradient.
    """

    theta = np.append(Theta1, Theta2).reshape(-1)

    numgrad = np.zeros_like(theta)
    perturb = np.zeros_like(theta)
    tol = 1e-4

    for p in range(len(theta)):
        # Set perturbation vector
        perturb[p] = tol
        loss1 = J(theta - perturb)
        loss2 = J(theta + perturb)

        # Compute numerical gradient
        numgrad[p] = (loss2 - loss1) / (2 * tol)
        perturb[p] = 0

    return numgrad

