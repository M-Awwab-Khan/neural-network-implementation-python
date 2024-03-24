import numpy as np

# activation function and its derivative
def tanh(x):
    return np.tanh(x)

def derivative_tanh(x):
    return 1-np.tanh(x)**2

def relu(x):
    return np.maximum(0, x)

def derivative_relu(x):
    return (x > 0).astype(int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivative_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))
  
