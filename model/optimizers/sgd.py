import numpy as np
from model.optimizers.optimizer import Optimizer
from model.layers.layer import Layer


class OptimizerSGD(Optimizer):
    def __init__(self, learning_rate=1., decay=0., momentum=0.):
        super().__init__(learning_rate, decay)

        self.momentum = momentum

    def update_params(self, layer: Layer):
        if self.momentum:
            if not hasattr(layer, 'weights_momentum'):
                layer.weights_momentum = np.zeros_like(layer.weights)
                layer.biases_momentum = np.zeros_like(layer.biases)

            weights_update = self.momentum * layer.weights_momentum - self.current_learning_rate * layer.weights_prime
            layer.weights_momentum = weights_update

            biases_update = self.momentum * layer.biases_momentum - self.current_learning_rate * layer.biases_prime
            layer.biases_momentum = biases_update

        else:
            weights_update = -self.current_learning_rate * layer.weights_prime
            biases_update = -self.current_learning_rate * layer.biases_prime

        layer.weights += weights_update
        layer.biases += biases_update
