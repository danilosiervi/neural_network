import numpy as np
from model.optimizers.optimizer import Optimizer
from model.layers.layer import Layer


class AdaGrad(Optimizer):
    def __init__(self, learning_rate=1., decay=0., epsilon=1e-7):
        super().__init__(learning_rate, decay)

        self.epsilon = epsilon

    def update_params(self, layer: Layer):
        if not hasattr(layer, 'weights_cache'):
            layer.weights_cache = np.zeros_like(layer.weights)
            layer.biases_cache = np.zeros_like(layer.biases)

        layer.weights_cache += layer.weights_prime ** 2
        layer.biases_cache += layer.biases_prime ** 2

        layer.weights += -self.current_learning_rate * layer.weights_prime / (np.sqrt(layer.weights_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.biases_prime / (np.sqrt(layer.biases_cache) + self.epsilon)
