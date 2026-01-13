import numpy as np

from model.losses.losses import get_loss

from model.losses.binary_crossentropy import BinaryCrossentropy
from model.losses.categorical_crossentropy import CategoricalCrossentropy
from model.losses.mean_squared_error import MeanSquaredError


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

    def train(self, x, y, *, epochs=10000, print_every=100, validation_data=None, patience=None, min_delta=0.0001):
        if validation_data is not None:
            x_val, y_val = validation_data

        if patience is not None:
            best_loss = float('inf')
            patience_counter = 0
            best_weights = None

        for epoch in range(epochs + 1):
            outputs = self.forward(x)

            if not epoch % print_every:
                if validation_data is not None:
                    outputs = self.forward(x_val)

                    data_loss = self.loss.calculate(outputs, y_val)
                    reg_loss = sum(self.loss.regularization_loss(layer) for layer in self.layers)
                    loss = data_loss + reg_loss

                    acc = self.evaluate(x_val, y_val)
                    outputs = self.forward(x)
                else:
                    data_loss = self.loss.calculate(outputs, y)
                    regularization_loss = sum(self.loss.regularization_loss(layer) for layer in self.layers)
                    loss = data_loss + regularization_loss

                    acc = self.evaluate(x, y)

                print(f'epoch: {epoch}, loss: {loss:.5f}, acc: {acc:.2f}%, lr: {self.optimizer.current_learning_rate:.10f}')

                if patience is not None:
                    if loss < best_loss - min_delta:
                        best_loss = loss
                        patience_counter = 0
                        best_weights = self._save_weights()
                    else:
                        patience_counter += print_every

                    if patience_counter >= patience:
                        print(f'\nEarly stopping triggered at epoch {epoch}')
                        print(f'Best loss: {best_loss:.5f}')
                        self._restore_weights(best_weights)
                        break

            self.backward(outputs, y)

            self.optimizer.pre_update_params()
            for layer in self.layers:
                self.optimizer.update_params(layer)
            self.optimizer.post_update_params()

    def _save_weights(self):
        weights = []
        for layer in self.layers:
            if hasattr(layer, 'weights') and hasattr(layer, 'biases'):
                weights.append({
                    'weights': layer.weights.copy(),
                    'biases': layer.biases.copy()
                })
            else:
                weights.append(None)
        return weights

    def _restore_weights(self, saved_weights):
        for layer, saved in zip(self.layers, saved_weights):
            if saved is not None:
                layer.weights = saved['weights'].copy()
                layer.biases = saved['biases'].copy()

    def evaluate(self, x, y):
        outputs = self.forward(x)
        accuracy = 0

        if isinstance(self.loss, CategoricalCrossentropy):
            predicted_classes = self.output_layer.activation.predictions(outputs)

            if len(y.shape) == 2:
                y = np.argmax(y, axis=1)

            accuracy = np.mean(predicted_classes == y)

        elif isinstance(self.loss, BinaryCrossentropy):
            predicted_classes = self.output_layer.activation.predictions(outputs)
            accuracy = np.mean(predicted_classes == y)

        elif isinstance(self.loss, MeanSquaredError):
            predictions = outputs

            y_true = y.reshape(-1, 1) if len(y.shape) == 1 else y
            preds = predictions.reshape(-1, 1)

            ss_res = np.sum((y_true - preds) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            accuracy = max(0, 1 - ss_res / ss_tot) if ss_tot != 0 else 0

        return accuracy * 100
