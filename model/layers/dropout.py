import numpy as np
from model.layers.layer import Layer


class DropoutLayer(Layer):
    def __init__(self, rate: float):
        self.inputs = np.array([])
        self.outputs = np.array([])
        self.inputs_prime = np.array([])
        self.outputs_prime = np.array([])

        self.binary_mask = np.array([])
        self.rate = 1 - rate

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        self.outputs = inputs * self.binary_mask

        return self.outputs

    def backward(self, outputs_prime: np.ndarray) -> np.ndarray:
        self.outputs_prime = outputs_prime.copy()
        self.inputs_prime = outputs_prime * self.binary_mask

        return self.inputs_prime
