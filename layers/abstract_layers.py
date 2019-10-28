from abc import ABC, abstractmethod

class Layer(ABC):
    _type = ""
    name = ""
    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        return self.__class__.__name__

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def compute_gradients(self, d):
        pass

class Activation(Layer):
    _type = "activation"

    def __init__(self, use_cache=True):
       self.cache = None
       self.use_cache = use_cache

    @abstractmethod
    def compute_gradients(self, d, cache=None):
        if self.use_cache and not cache:
            cache = self.cache
        return cache

class TrainableLayer(Layer):

    _type = "trainable"


    @abstractmethod
    def __init__(self, trainable):
       self.trainable = trainable

    def _initialize(self):
        pass

    @abstractmethod
    def apply_gradients(self, optimizer, **kwargs):
        pass

