import numpy as np

from layers.abstract_layers import Activation


class ReLU(Activation):
    def forward(self, x, **kwargs):
        if self.use_cache:
            self.cache = x
        return np.maximum(0, x)

    def compute_gradients(self, d=1, cache=None):
        cache = super().compute_gradients(d, cache)
        dx = d * (cache > 0)
        return dx


class SoftMax(Activation):

    def forward(self, x, **kwargs):
        out = x - np.max(x, axis=1, keepdims=True) # avoid numeric instability
        out = np.exp(out)
        out = out / np.sum(out, axis=1, keepdims=True)
        if self.use_cache:
            self.cache = x
        return out

    def compute_gradients(self, d=1, cache=None): #?
        cache = super().compute_gradients(d, cache)
        # ????
        return d # see cross entropy error


class Tanh(Activation):
    def forward(self, x, **kwargs):
        out = np.tanh(x)
        if self.use_cache:
            self.cache = out
        return out

    def compute_gradients(self, d=1, cache=None):
        cache = super().compute_gradients(d, cache)
        dx = d*(1-cache**2)
        return dx

class Sigmoid(Activation):
    def forward(self, x, **kwargs):
        x = np.clip(x, -700, 1000)
        out = 1/(1+np.exp(-x))
        if self.use_cache:
            self.cache = out
        return out

    def compute_gradients(self, d=1, cache=None):
        cache = super().compute_gradients(d, cache)
        dx = d*(1-cache)*cache
        return dx

class HardSigmoid(Activation):
    def forward(self, x, **kwargs):
        out = (0.2 * x) + 0.5
        out = np.clip(out, 0, 1)
        if self.use_cache:
            self.cache = out
        return out

    def compute_gradients(self, d=1, cache=None):
        cache = super().compute_gradients(d, cache)
        eps = 1e-6
        dx = ((cache > eps) & (cache < 1-eps)) * 0.2 * d
        return dx




softmax = SoftMax
relu = ReLU
tanh = Tanh
sigmoid = Sigmoid
hard_sigmoid = HardSigmoid
