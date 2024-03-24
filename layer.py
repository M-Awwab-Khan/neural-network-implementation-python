#Base Class Layer

class Layer:

    def __init__(self):
        self.input = None
        self.output = None

    # computes the output Y of a layer for a given input X
    def forward_propagate(self, input):
        raise NotImplementedError

     # computes dE/dX for a given dE/dY (and update parameters if any)
    def backward_propagate(self, output_error, learning_rate):
        raise NotImplementedError
