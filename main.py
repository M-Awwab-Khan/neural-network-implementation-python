import numpy as np

from network import Network
from fclayer import FCLayer
from activation_layer import ActivationLayer
from activations import tanh, derivative_tanh
from loss_function import mse, derivative_mse

# training data
x_train = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y_train = np.array([
    [0],
    [1],
    [1],
    [0]
])

# network
net = Network()
net.add(FCLayer(2, 3))
net.add(ActivationLayer(tanh, derivative_tanh))
net.add(FCLayer(3, 1))
net.add(ActivationLayer(tanh, derivative_tanh))

# train
net.use(mse, derivative_mse)
net.fit(x_train, y_train, epochs=1000, learning_rate=0.1)

# test
out = net.predict(x_train)
print(out)
