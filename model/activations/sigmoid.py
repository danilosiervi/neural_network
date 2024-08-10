import numpy as np
from model.activations.activation import ActivationFunction


class Sigmoid(ActivationFunction):
    def __init__(self):
        def activation_function(inputs: np.ndarray) -> np.ndarray:
            self.outputs = 1 / (1 + np.exp(-inputs))
            return self.outputs

        def activation_function_prime(outputs_prime: np.ndarray) -> np.ndarray:
            self.inputs_prime = outputs_prime * self.outputs * (1 - self.outputs)
            return self.inputs_prime

        def predictions(outputs: np.ndarray) -> np.ndarray:
            return (outputs > 0.5) * 1

        super().__init__(activation_function, activation_function_prime, predictions)
