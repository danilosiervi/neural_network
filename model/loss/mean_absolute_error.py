import numpy as np
from model.loss.loss import Loss


class MeanAbsoluteError(Loss):
    def forward(self, y_pred, y_true) -> np.ndarray:
        sample_losses = np.mean(np.abs(y_true - y_pred), axis=1)
        return sample_losses

    def backward(self, outputs_prime, y_true) -> np.ndarray:
        samples = len(outputs_prime)
        outputs = len(outputs_prime[0])

        inputs_prime = np.sign(y_true - outputs_prime) / outputs
        inputs_prime /= samples

        return inputs_prime
