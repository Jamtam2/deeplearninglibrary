"""
Our neural nets will be made up of layers
Each layer needs to passs its input forward 
and propagate gradients backward. For example,
a neural net might look like

inputs -> Linear -> Tanh -> Linear -> output
"""
from tensor import Tensor

from typing import Dict, Callable
import numpy as np 


class Layer:
    def __init__(self) -> None:
        self.params: Dict[str, Tensor] = { }
        self.grads: Dict[str, Tensor] = { }

    def forward(self,inputs:Tensor) ->Tensor:
        """
        Produce the outputs corresponding to these inputs
        """
        raise NotImplementedError
    def backward(self, grad: Tensor) -> Tensor:
        """
        Backpropagate this gradient through the layer
        """
        raise NotImplementedError
    
class Linear(Layer):
    """
    computers output = inputs @ w + b
    
    matrix multiplied by a weight plus bias 
    """
    def __init__(self, input_size: int, output_size: int) -> None:
        #inputs will be (batch_size, input_size)
        #outputs will be (batch_size, output_size)
        super().__init__()
        self.params['w'] = np.random.rand(input_size,output_size)
        self.params['b'] = np.random.rand(output_size)
    def forward(self,inputs:Tensor) ->Tensor:
        """
        outputs = inputs @ w + b
        """
        self.inputs = inputs
        return inputs @ self.params["w"] + self.params["b"]
    
    
    def backward(self, grad: Tensor) -> Tensor:
        """
        A derivative is the measure of how a function changes as it's input changes:
            -if i nudge input a tiny bit, how much does my output change?

        Partial derivative is when a function depends on more than one variable
            -want to know how the function changes with respect to one of these variables; keeping the others constant
            

        if y = f(x) and x = a * b + c
        then dy/da = f'(x) * b
        and dy/db = f'(x) * a
        and dy/dc = f'(x)
        
        if y = f(x) and x = a @ b + c

        then dy/da = f'(x) @ b.T (transpose)
        and dy/db = f'(x) @ a.T (transpose
        and dy/dc = f'(x) 
        """
        self.grads['b'] = np.sum(grad, axis=0)
        self.grads['w'] = self.inputs.T @ grad
        return grad @ self.params['w'].T

F = Callable[[Tensor], Tensor]

class Activation(Layer):
    """
    F takes a list of tensors from input and the output of a single tensor
    returns a tensor
    An activation layer just applies a functino elementwise to its inputs
    """

    def __init__(self,f: F, f_prime: F) -> None:
        super().__init__()
        self.f = f
        self.f_prime = f_prime
    def forward(self, inputs:Tensor) -> Tensor:
        self.inputs = inputs
        return self.f(inputs)
    def backward(self, grad: Tensor) -> Tensor:
        """
        f is ther rest of the neural net
        g is the part being done by this layer

        g(z) is f' of the inputs
        f'(x) is the gradient with respect to the output of this
        if y = f(x) and x = g(z)
        then dy/dz = f'(x) * g'(z)
        """
        return self.f_prime(self.inputs) * grad
    

def tanh(x: Tensor) -> Tensor:
    return np.tanh(x)

#just the derivative of the tanh function
def tanh_prime(x: Tensor) ->Tensor:
    y = tanh(x)
    return 1- y ** 2

class Tanh(Activation):
    def __init__(self):
        super().__init__(tanh,tanh_prime)