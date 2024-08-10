import numpy as np
from model.activations.activation import ActivationFunction


class Softmax(ActivationFunction):
    def __init__(self):
        def activation_function(inputs: np.ndarray) -> np.ndarray:
            exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
            self.outputs = exp_values / np.sum(exp_values, axis=1, keepdims=True)

            return self.outputs

        def activation_function_prime(outputs_prime: np.ndarray) -> np.ndarray:
            self.inputs_prime = np.empty_like(outputs_prime)

            for index, (single_output, single_output_prime) in enumerate(zip(self.outputs, outputs_prime)):
                single_output = single_output.reshape(-1, 1)
                jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)

                self.inputs_prime[index] = np.dot(jacobian_matrix, single_output_prime)

            return self.inputs_prime

        def predictions(outputs: np.ndarray) -> np.ndarray:
            return np.argmax(outputs, axis=1)

        super().__init__(activation_function, activation_function_prime, predictions)

    def loss_backward(self, outputs_prime, y_true):
        samples = len(outputs_prime)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.inputs_prime = outputs_prime.copy()
        self.inputs_prime[range(samples), y_true] -= 1
        self.inputs_prime /= samples

        return self.inputs_prime
