import numpy as np
import numba

class SGD:

    def __init__(self, learning_rate=1e-1):
        self.learning_rate = learning_rate

    def __call__(self, weights, gradients, learning_rate = None):
        self.apply_gradients(weights, gradients, learning_rate)

    def apply_gradients(self, weights, gradients, learning_rate = None):
        if not learning_rate:
            learning_rate = self.learning_rate
        weights -= gradients*learning_rate


class Momentum(SGD):

    def __init__(self, learning_rate=1e-2, momentum=0.9):
        super().__init__(learning_rate)

        self.momentum = momentum
        self.velocities = {}


    def apply_gradients(self, weights, gradients, learning_rate = None):
        w_id = id(weights)
        if not learning_rate:
            learning_rate = self.learning_rate

        if w_id not in self.velocities:
            self.velocities[w_id] = np.zeros_like(weights)

        self.velocities[w_id] = self.momentum*self.velocities[w_id] - learning_rate*gradients
        weights += self.velocities[w_id]


class Nesterov(Momentum):
    def __init__(self, learning_rate=1e-2, momentum=0.9):
        super().__init__(learning_rate, momentum)

    def apply_gradients(self, weights, gradients, learning_rate = None):
        w_id = id(weights)
        if not learning_rate:
            learning_rate = self.learning_rate

        if w_id not in self.velocities:
            self.velocities[w_id] = np.zeros_like(weights)

        v_prev = self.velocities[w_id]
        self.velocities[w_id] = self.momentum*self.velocities[w_id] - learning_rate*gradients
        weights += -self.momentum*v_prev + (1+self.momentum)*self.velocities[w_id]


class RMSprop(Momentum):

    def __init__(self, learning_rate=1e-3, momentum=0.0, decay_rate=0.99, epsilon=1e-8):
        super().__init__(learning_rate, momentum)
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.mean_squares = {}

    def apply_gradients(self, weights, gradients, learning_rate = None):
        w_id = id(weights)
        if not learning_rate:
            learning_rate = self.learning_rate

        if w_id not in self.mean_squares:
            self.mean_squares[w_id] = np.zeros_like(weights)

        if w_id not in self.velocities:
            self.velocities[w_id] = np.zeros_like(weights)

        self.mean_squares[w_id] = self.decay_rate * self.mean_squares[w_id] + (1 - self.decay_rate) * (gradients**2)        # exponential weighted average

        self.velocities[w_id] = self.momentum*self.velocities[w_id] - learning_rate*gradients/(np.sqrt(self.mean_squares[w_id]) + self.epsilon)

        weights += self.velocities[w_id]

class Adam(RMSprop):

    def __init__(self, learning_rate=1e-3, momentum=0.9, decay_rate=0.999, epsilon=1e-8):
        super().__init__(learning_rate, momentum, decay_rate, epsilon)
        self.t = {}

    def apply_gradients(self, weights, gradients, learning_rate = None):
        w_id = id(weights)

        if not learning_rate:
            learning_rate = self.learning_rate

        if w_id not in self.mean_squares:
            self.mean_squares[w_id] = np.zeros_like(weights)

        if w_id not in self.velocities:
            #print("w20", weights.shape)
            self.velocities[w_id] = np.zeros_like(weights)
            #print('v20', self.velocities[w_id].shape)
        if w_id not in self.t:
            self.t[w_id] = 0


        self.t[w_id] += 1

        #print('grads', gradients)
        #print("vt", self.velocities[w_id])
        self.velocities[w_id] =  self.momentum * self.velocities[w_id] + (1 - self.momentum) * gradients # momentum
        #print("vt", self.velocities[w_id].shape)
        velocities = self.velocities[w_id] / (1 - self.momentum**self.t[w_id]) # bias correction
        self.mean_squares[w_id] = self.decay_rate * self.mean_squares[w_id] + (1 - self.decay_rate) * (gradients**2) # RMSprop
        mean_squares = self.mean_squares[w_id] / (1 - self.decay_rate**self.t[w_id]) # bias correction
        #print("w", weights.shape)
        #print("v", velocities.shape)
        weights -= self.learning_rate * velocities / (np.sqrt(mean_squares) + self.epsilon)

