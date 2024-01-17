""""
A loss functino measures how good our predictions are, we can use this to adjust parameters of our network
"""

import numpy as np

from tensor import Tensor

class Loss:
    #abstract base loss class
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError
    #gradient, vector matrix of partial derivatives of loss function
    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError
    
class MSE(Loss):
    """
    MSE is mean squared error, although we're just doing total squared error
    """
    #error is just predicted minus actual
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        return np.sum((predicted - actual) **2)
        
    #derivate of above; 2 * x 
    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return 2 * (predicted-actual)
        