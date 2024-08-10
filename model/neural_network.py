import numpy as np


class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.trainable_layers = []

        self.optimizer = None
        self.loss = None
        self.accuracy = None

        self.input_layer = None
        self.output_layer_activation = None

        self.softmax_classifier_output = None

    def add(self, layer):
        self.layers.append(layer)

    def set(self, *, loss, optimizer, accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    def finalize(self):
        self.input_layer = InputLayer()
        self.trainable_layers = []

        layer_count = len(self.layers)

        for i in range(layer_count):
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i + 1]

            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.layers[i + 1]

            else:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])

            self.loss.remember_trainable_layers(
                self.trainable_layers
            )

    def forward(self, x, training):
        self.input_layer.forward(x, training)

        for layer in self.layers:
            layer.forward(layer.next.outputs)

        return layer.output

    def backward(self, output, y):
        if isinstance(self.layers[-1], ActivationSoftmax) and isinstance(self.loss, CategoricalCrossentropy):
            self.layers[-1].loss_backward(y, output)

        self.loss.backward(output, y)

        for layer in reversed(self.layers):
            layer.backward(layer.next.inputs_prime)

    def train(self, x, y, *, epochs=10000, print_every=100, validation_data=None):
        self.accuracy.init(y)

        for epoch in range(1, epochs + 1):
            output = self.forward(x)

            data_loss, regularization_loss = self.loss.calculate(output, y, include_regularization=True)
            loss = data_loss + regularization_loss

            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y)

            self.backward(output, y)

            self.optimizer.pre_update_params()
            for layer in self.trainable_layers:
                self.optimizer.update_params(layer)
            self.optimizer.post_update_params()

            if not epoch % print_every:
                print(f'epoch: {epoch}: '
                      f'loss: {loss:.3f}, '
                      f'accuracy: {accuracy:.3f}, '
                      f'lr: {optimizer.current_learning_rate:.5f}')

        if validation_data is not None:
            x_val, y_val = validation_data

            output = self.forward(x_val, training=False)
