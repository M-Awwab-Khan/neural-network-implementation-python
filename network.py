class Network:

    def __init__(self):
        self.layers = []
        self.loss = None
        self.derivative_loss = None

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    # set loss to use
    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime
