import numpy as np
import matplotlib.pyplot as plt
from utils import load_data, load_weights,one_hot_encoding, accuracy, displayData
from MLP import MLP
from public_test import compute_cost_test, predict_test, confusion_matrix_binary


x,y = load_data('data/ex3data1.mat')
theta1, theta2 = load_weights('data/ex3weights.mat')

#TO-DO: calculate a testing a prediction and cost.
#Ejercicio 1
mlp = MLP(theta1, theta2)
a1, a2, a3, z2, z3 = mlp.feedforward(x)

#Ejercicio 2
p = mlp.predict(a3)
predict_test(p,y, accuracy)
y_one_hot = one_hot_encoding(y)

compute_cost_test(mlp, a3,y_one_hot )

#Ejercicio 3
metrics = confusion_matrix_binary(y, p, positive_class=0)
indx_p_Cero = np.where(p == 9)[0][:25]
displayData(x[indx_p_Cero, :])
plt.show()

