from layer import Layer
import numpy as np

# inherit from base class Layer 
class FCLayer(Layer):

    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(input_size, output_size) / np.sqrt(input_size + output_size)
        self.bias = np.random.randn(1, output_size) / np.sqrt(input_size + output_size)

    # returns output for a given input
    def forward_propagate(self, input_data):
        self.input = input_data
        self.output = np.dot(input_data, self.weights) + self.bias

        return self.output

    def backward_propagate(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.reshape(1, -1).T, output_error)
        bias_error = output_error

        # updating parameters
        self.weights = self.weights - learning_rate * weights_error
        self.bias = self.bias - learning_rate * bias_error

        return input_error
