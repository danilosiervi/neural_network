import numpy as np
from model.activations.activation import ActivationFunction


class Relu(ActivationFunction):
    def __init__(self):
        def activation_function(inputs: np.ndarray) -> np.ndarray:
            self.outputs = np.maximum(0, inputs)
            return self.outputs

        def activation_function_prime(outputs_prime: np.ndarray) -> np.ndarray:
            self.inputs_prime = outputs_prime.copy()
            self.inputs_prime[self.inputs <= 0] = 0

            return self.inputs_prime

        def predictions(outputs: np.ndarray) -> np.ndarray:
            return outputs

        super().__init__(activation_function, activation_function_prime, predictions)
