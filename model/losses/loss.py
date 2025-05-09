import numpy as np


class Loss:
    def calculate(self, output, y) -> float:
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)

        return data_loss

    def regularization_loss(self, layer) -> float:
        regularization_loss = 0.

        if not hasattr(layer, 'weights'):
            return 0.

        if layer.weight_regularizer_l1 > 0:
            regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
        if layer.weight_regularizer_l2 > 0:
            regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)

        if layer.bias_regularizer_l1 > 0:
            regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))
        if layer.bias_regularizer_l2 > 0:
            regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)

        return regularization_loss
