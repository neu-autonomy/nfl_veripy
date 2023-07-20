class Propagator:
    def __init__(self, input_shape=None):
        self.input_shape = input_shape

    @property
    def network(self):
        return self._network

    @network.setter
    def network(self, network):
        self._network = self.torch2network(network)
