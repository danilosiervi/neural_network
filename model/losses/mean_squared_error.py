import numpy as np
from model.losses.loss import Loss


class MeanSquaredError(Loss):
    def forward(self, y_pred, y_true):
        sample_losses = np.mean((y_true - y_pred) ** 2, axis=1)
        return sample_losses

    def backward(self, outputs_prime, y_true):
        samples = len(outputs_prime)
        outputs = len(outputs_prime[0])

        inputs_prime = -2 * (y_true - outputs_prime) / outputs
        inputs_prime /= samples

        return inputs_prime
