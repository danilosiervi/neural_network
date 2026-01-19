import numpy as np
from model.layers.layer import Layer
from model.activations.activation import ActivationFunction
from model.activations.activations import get_activation


class DenseLayer(Layer):
    def __init__(
            self,
            input_shape,
            n_neurons,
            activation=None,
            weight_regularizer_l1=0.,
            weight_regularizer_l2=0.,
            bias_regularizer_l1=0.,
            bias_regularizer_l2=0.
    ):
        self.inputs = np.array([])
        if activation == 'Relu':
            self.weights = np.random.randn(input_shape, n_neurons) * np.sqrt(2 / input_shape)
        else:
            self.weights = 0.01 * np.random.randn(input_shape, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.outputs = np.array([])

        self.inputs_prime = np.array([])
        self.weights_prime = np.array([])
        self.biases_prime = np.array([])
        self.outputs_prime = np.array([])

        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

        self.activation = get_activation(activation)()

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs.copy()
        self.outputs = np.dot(inputs, self.weights) + self.biases

        return self.activation.forward(self.outputs)

    def backward(self, activation_outputs_prime: np.ndarray, skip_activation: bool = False) -> np.ndarray:
        if skip_activation:
            outputs_prime = activation_outputs_prime
        else:
            outputs_prime = self.activation.backward(activation_outputs_prime)

        self.outputs_prime = outputs_prime.copy()
        self.weights_prime = np.dot(self.inputs.T, outputs_prime)
        self.biases_prime = np.sum(outputs_prime, axis=0, keepdims=True)

        if self.weight_regularizer_l1 > 0:
            dl1 = np.ones_like(self.weights)
            dl1[self.weights < 0] = -1
            self.weights_prime += self.weight_regularizer_l1 * dl1
        if self.weight_regularizer_l2 > 0:
            self.weights_prime += 2 * self.weight_regularizer_l2 * self.weights

        if self.bias_regularizer_l1 > 0:
            dl1 = np.ones_like(self.biases)
            dl1[self.biases < 0] = -1
            self.biases_prime += self.bias_regularizer_l1 * dl1
        if self.bias_regularizer_l2 > 0:
            self.biases_prime += 2 * self.bias_regularizer_l2 * self.biases

        self.inputs_prime = np.dot(outputs_prime, self.weights.T)

        return self.inputs_prime

    def get_parameters(self):
        return self.weights, self.biases

    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases
