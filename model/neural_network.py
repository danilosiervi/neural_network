import numpy as np


class NeuralNetwork:
    def __init__(self, layers=np.ndarray([])):
        self.layers = []
        self.trainable_layers = []

        for i in layers:
            self.layers.append(i)

        self.optimizer = None
        self.loss = None
        self.accuracy = None

        self.input_layer = None
        self.output_layer = None

    def add(self, layer):
        self.layers.append(layer)

    def set(self, *, loss, optimizer, accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    def finalize(self):
        pass

    def forward(self, x, training):
        self.input_layer.forward(x, training)

        for layer in self.layers:
            layer.forward(layer.next.outputs)

        return layer.output

    def backward(self, output, y):
        pass

    def train(self, x, y, *, epochs=10000, print_every=100, validation_data=None):
        pass
