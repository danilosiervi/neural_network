import numpy as np


class Optimizer:
    def __init__(self, learning_rate, decay):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def post_update_params(self):
        self.iterations += 1
