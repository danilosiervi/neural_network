import numpy as np


class ActivationFunction:
    def __init__(self, activation_function, activation_function_prime, predictions):
        self.inputs = np.array([])
        self.outputs = np.array([])
        self.inputs_prime = np.array([])
        self.outputs_prime = np.array([])

        self._activation_function = activation_function
        self._activation_function_prime = activation_function_prime
        self.predictions = predictions

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs.copy()
        return self._activation_function(inputs)

    def backward(self, outputs_prime: np.ndarray) -> np.ndarray:
        self.outputs_prime = outputs_prime.copy()
        return self._activation_function_prime(self.outputs_prime)

    def predictions(self) -> np.ndarray:
        return self.predictions(self.outputs)
