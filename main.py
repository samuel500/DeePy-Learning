import numpy as np
import os
import random
from time import time

from layers.activations import *
from layers.layers import *
from initializers import *
from utils.data import *
from costs import *
from layers import *
from optim import *
from models import *


from vae import *
from rnn import *



def main():
    data = load_mnist()
    T = 3
    #data0 = next(data)
    data = list(data)
    #data = [(y, T*[x.flatten()]) for y,x in data]

    #data = [(y, x.T) for y, x in data]
    #print(data[0])
    '''
    layers = [
            #GRULayer(128),
            LSTMLayer(256),
            #RNNLayer(128),
            FC(10),
            SoftMax()
        ]
    '''
    layers = [
        FC(512),
        ReLU(),
        FC(32),
        ReLU(),
        Batchnorm(trainable=False),
        FC(512),
        ReLU(),
        FC(784),
        Sigmoid()
    ]

    #nn_mnist = Classifier(data[0][1].shape, 10, layers=layers)
    #nn_mnist = Classifier((784, T), 10, layers=layers)
    #nn_mnist = AE(784, 784, layers=layers)

    bs = 128

    # test_data

    test_data = list(load_mnist(dataset='testing'))
    test_data = list(zip(*test_data))
    testX, testY = test_data[1], one_hot(list(test_data[0]))
    #testX = [T*[x.flatten()] for x in testX]

    testX = np.array(testX)

    #testX = [t.T for t in testX]
    #testX = np.array(testX)

    vae_mn = VAE(data[0][1].shape, 32, 784)
    vae_mn.train(data, 100, testX/255, testY, bs, test_rate=1)

    #print(testX.shape)
    #nn_mnist.test(testX/255, testY)
    #nn_mnist.train(data, 100, testX/255, testY, bs)
    #testX = testX.reshape(len(testX), 784)




    """

    #vae_mn = VAE(data[0][1].shape, 32, 784)
    #vae_mn.train(data, 100, testX/255, testY, bs, test_rate=1)
    #vae_mn.latent2d2(testX/255, testY)

    """


    #nn_mnist.plot_loss()

    #nn_mnist.test(testX/255, testX/255)

    #nn_mnist0 = AE(data0[1].shape, 784)
    #nn_mnist0.train(data, 40, testX/255, testX/255, bs, test_rate=50)
    #nn_mnist0.plot_loss()


    """

    l_1 = [
        FC(512, trainable=True),
        ReLU(),
        FC(128, trainable=True),
        ReLU(),
        FC(512, trainable=True),
        ReLU(),
        #FC(128, trainable=False),
        #ReLU(),
        FC(784, trainable=True),
        sigmoid()
    ]
    nn_mnist1 = AE(data0[1].shape, 784, layers=l_1)
    nn_mnist1.train(data[:20000], 40, testX/255, testX/255, bs, test_rate=50)

    l_2 = [
        FC(512, trainable=True),
        ReLU(),
        FC(256, trainable=True),
        ReLU(),
        FC(64, trainable=True),
        ReLU(),
        FC(256, trainable=True),
        ReLU(),
        FC(512, trainable=True),
        ReLU(),
        #FC(128, trainable=False),
        #ReLU(),
        FC(784, trainable=True),
        sigmoid()
    ]
    nn_mnist2 = AE(data0[1].shape, 784, layers=l_2)
    nn_mnist2.train(data[20000:40000], 40, testX/255, testX/255, bs, test_rate=50)

    l_3 = [
        FC(1024, trainable=True),
        ReLU(),
        FC(64, trainable=True),
        ReLU(),
        FC(1024, trainable=True),
        ReLU(),
        #FC(128, trainable=False),
        #ReLU(),
        FC(784, trainable=True),
        sigmoid()
    ]
    nn_mnist3 = AE(data0[1].shape, 784, layers=l_3)
    nn_mnist3.train(data[-30000:], 40, testX/255, testX/255, bs, test_rate=50)


    input_mods = [
        {'model': nn_mnist0, 'layer': 1},
        {'model': nn_mnist1, 'layer': 1},
        {'model': nn_mnist2, 'layer': 2},
        {'model': nn_mnist3, 'layer': 1},
    ]

    print('Ensemble')

    ensemble = Ensemble(data0[1].shape, 10, input_models=input_mods)
    ensemble.train(data, 100, testX/255, testY, bs)
    """
if __name__=='__main__':
    main()
