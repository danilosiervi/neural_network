import copy

import numpy as np
from model.losses.losses import get_loss
from model.activations.softmax import Softmax
from model.losses.categorical_crossentropy import CategoricalCrossentropy


class NeuralNetwork:
    def __init__(self, layers=None):
        if layers is None:
            layers = []
        self.layers = layers

        self.input_layer = layers[0]
        self.output_layer = layers[-1]
        self.hidden_layers = layers[1:-1]

        self.optimizer = None
        self.loss = None
        self.accuracy = None

    def add(self, layer):
        self.layers.append(layer)

    def set(self, *, loss, optimizer):
        self.loss = get_loss(loss)()
        self.optimizer = optimizer

    def forward(self, x):
        outputs = self.input_layer.forward(x)

        for layer in self.hidden_layers:
            outputs = layer.forward(outputs)

        return self.output_layer.forward(outputs)

    def backward(self, outputs, y):
        loss_outputs = self.loss.backward(outputs, y)
        inputs_prime = self.output_layer.backward(loss_outputs)

        for layer in reversed(self.hidden_layers):
            inputs_prime = layer.backward(inputs_prime)

        self.input_layer.backward(inputs_prime)

    def train(self, x, y, *, epochs=10000, print_every=100):
        for epoch in range(epochs + 1):
            outputs = self.forward(x)

            data_loss = self.loss.calculate(outputs, y)
            regularization_loss = 0.

            for layer in self.layers:
                regularization_loss += self.loss.regularization_loss(layer)

            loss = data_loss + regularization_loss

            if not epoch % print_every:
                print(f'epoch: {epoch}, '
                      f'loss: {loss:.3f}, '
                      f'lr: {self.optimizer.current_learning_rate:.10f}')

            self.backward(outputs, y)

            self.optimizer.pre_update_params()
            for layer in self.layers:
                self.optimizer.update_params(layer)
            self.optimizer.post_update_params()
