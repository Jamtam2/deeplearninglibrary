"""
The canonical example of a function that can't be learned with a 
simple linear model is XOR, XOR is not linearly seperable
"""

from train import train
from nn import NeuralNet
from layers import Linear, Tanh
import numpy as np

inputs = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

targets = np.array([
    [1,0],
    [0,1],
    [0,1],
    [1,0]
])

net = NeuralNet([Linear(input_size=2,output_size=2), 
                Tanh(),
                Linear(input_size=2,output_size=2)])

train(net,inputs,targets)

for x,y in zip(inputs,targets):
    predicted = net.forward(x)
    print(x,predicted,y)
