import os
import gym
import random
from multiprocessing import Pool, Process, Pipe
from utils.data import *

import numpy as np

from layers.activations import *
from initializers import *

import copy

class Population:

    def __init__(self):
        self.eps = 0.01
        self.N = 1000
        self.T = 10

        self.agents = [Agent() for _ in range(self.N)]
        self.data = list(load_mnist())[:20000]

    def train(self):
        print(len(self.agents))
        for i in range(self.N):
            #if not i%500:
            #    print(i)
            x, y = rand_batch(self.data, batch_size=256)
            self.agents[i].evaluate(x, y)
        best = list(sorted(self.agents))[-self.T:]
        print(best)
        self.agents = copy.deepcopy(best)
        for y in range(99):
            new_best = copy.deepcopy(best)
            for b in new_best: b.mutated()
            #if y == 50: print(new_best[4].layers[0].W)
            self.agents += copy.deepcopy(new_best)
        #self.agents = [Agent()]


class Agent:

    def __init__(self, layers=None):
        if not layers:
            self.layers = [
                FCMem(128, (784)),
                ReLU(),
                FCMem(10, 128),
                SoftMax()
            ]
        else:
            self.layers = layers

        self.acc = 0

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def evaluate(self, x, y):
        preds = self.forward(x)
        self.acc = sum(np.argmax(preds, axis=1) == np.argmax(y, axis=1)) / len(y)

    def mutated(self):
        for layer in self.layers:
            if hasattr(layer, 'mutate'):
                layer.mutate()

    def __lt__(self, b):
        if self.acc < b.acc:
            return True
        else:
            return False
    def __str__(self):
        return str(self.acc)
    def __repr__(self):
        return str(self.acc)

class FC:

    def __init__(self, units, input_shape=None, use_bias=True, weight_initializer=np.random.normal, bias_initializer=np.zeros):
        self.units = units
        self.input_shape = input_shape
        self.use_bias = use_bias
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        s = np.random.randint(0, 50000)
        self.eps = 0.002
        self.seeds = [s]

    def forward(self, x):
        shape = (np.prod(self.input_shape), self.units)
        s = self.seeds[0]
        np.random.seed(s)
        W = self.weight_initializer(size=shape, scale=np.sqrt(2/shape[0]))
        np.random.seed(s)
        B = self.bias_initializer((self.units))
        for s in self.seeds[1:]:
            np.random.seed(s)
            W += self.weight_initializer(size=shape, scale=self.eps)
            np.random.seed(s)
            B += np.random.normal(size=(self.units), scale=self.eps)


        x = x.reshape(x.shape[0], np.prod(x.shape[1:])) # flatten input
        out = x.dot(W)
        if self.use_bias:
            out += B
        return out

    def mutate(self):
        s = random.randint(0, 50000)
        self.seeds.append(s)


class FCMem:

    def __init__(self, units, input_shape=None, use_bias=True, weight_initializer=np.random.normal, bias_initializer=np.zeros):
        self.units = units
        self.input_shape = input_shape
        self.use_bias = use_bias
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        s = np.random.randint(0, 50000)
        self.eps = 0.006
        self.seeds = [s]

        shape = (np.prod(self.input_shape), self.units)
        np.random.seed(s)
        self.W = self.weight_initializer(size=shape, scale=np.sqrt(2/shape[0]))
        np.random.seed(s)
        self.B = self.bias_initializer((self.units))

    def forward(self, x):

        x = x.reshape(x.shape[0], np.prod(x.shape[1:])) # flatten input
        out = x.dot(self.W)
        if self.use_bias:
            out += self.B
        return out

    def mutate(self):
        s = np.random.randint(0, 20000)
        self.seeds.append(s)
        shape = (np.prod(self.input_shape), self.units)
        np.random.seed(s)
        self.W += self.weight_initializer(size=shape, scale=self.eps)
        np.random.seed(s)
        self.B += np.random.normal(size=(self.units), scale=self.eps)

if __name__=='__main__':
    p = Population()
    for i in range(500):
        print(i)
        p.train()
