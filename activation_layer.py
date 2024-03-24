from layer import Layer 

# inherits from the base Layer class
class ActivationLayer(Layer):

    def __init__(self, activation, derivative_activation):
        self.activation = activation
        self.derivative_activation = derivative_activation

    # returns the activated input
    def forward_propagate(self, input_data):
        self.input = input_data
        self.ouput = self.activation(self.input)

        return self.output

    # Returns input_error=dE/dX for a given output_error=dE/dY.
    # learning_rate is not used because there is no "learnable" parameters.
    def backward_propagate(self, output_error, learning_rate):
        return self.derivative_activation(self.input) * output_error
