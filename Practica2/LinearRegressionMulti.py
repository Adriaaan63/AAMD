import numpy as np
import copy
import math

from LinearRegression import LinearReg

class LinearRegMulti(LinearReg):

    """
    Computes the cost function for linear regression.

    Args:
        x (ndarray): Shape (m,) Input to the model
        y (ndarray): Shape (m,) the real values of the prediction
        w, b (scalar): Parameters of the model
        lambda: Regularization parameter. Most be between 0..1. 
        Determinate the weight of the regularization.
    """
    def __init__(self, x, y,w,b, lambda_):
        super().__init__(x, y, w, b)
        self.lambda_ = lambda_
        return

    def f_w_b(self, x):
        ret = x @ self.w + self.b
        return ret

    
    """
    Compute the regularization cost (is private method: start with _ )
    This method will be reuse in the future.

    Returns
        _regularizationL2Cost (float): the regularization value of the current model
    """
    
    def _regularizationL2Cost(self):
        
        m = len(self.x)

        reg_cost = (self.lambda_ / (2 * m)) * np.sum(self.w ** 2)
        return reg_cost
    
    """
    Compute the regularization gradient (is private method: start with _ )
    This method will be reuse in the future.

    Returns
        _regularizationL2Gradient (vector size n): the regularization gradient of the current model
    """ 
    
    def _regularizationL2Gradient(self):
        m = len(self.x)
        reg_grad = (self.lambda_ / m) * self.w 
        return reg_grad

    def compute_cost(self):
        m = len(self.x)
        y_pred = self.f_w_b(self.x)

        total_cost = (1 / (2 * m)) * np.sum((self.y - y_pred) ** 2)
        total_cost += self._regularizationL2Cost()
        return np.float64(total_cost)
    
    def compute_gradient(self):
       
        m = len(self.x)
        y_pred = self.f_w_b( self.x)

        err = y_pred - self.y

        dj_dw = (1 / m) * (self.x.T @ err) + self._regularizationL2Gradient()
        dj_db = (1 / m) * np.sum(err)

        return np.array(dj_dw, dtype=np.float64), np.float64(dj_db)



def cost_test_multi_obj(x,y,w_init,b_init):
    lr = LinearRegMulti(x,y,w_init,b_init,0)
    cost = lr.compute_cost()
    return cost

def compute_gradient_multi_obj(x,y,w_init,b_init):
    lr = LinearRegMulti(x,y,w_init,b_init,0)
    dw,db = lr.compute_gradient()
    return dw,db
