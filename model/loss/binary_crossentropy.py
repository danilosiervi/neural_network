import numpy as np
from model.loss.loss import Loss


class BinaryCrossentropy(Loss):
    def forward(self, y_pred, y_true) -> np.ndarray:
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        sample_losses = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=1)

        return sample_losses

    def backward(self, outputs_prime, y_true) -> np.ndarray:
        samples = len(outputs_prime)
        outputs = len(outputs_prime[0])
        clipped_outputs_prime = np.clip(outputs_prime, 1e-7, 1 - 1e-7)

        inputs_prime = -(y_true / clipped_outputs_prime - (1 - y_true) / (1 - clipped_outputs_prime)) / outputs
        inputs_prime /= samples

        return inputs_prime
