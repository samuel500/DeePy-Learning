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
    data = list(data)



    # test_data

    test_data = list(load_mnist(dataset='testing'))
    test_data = list(zip(*test_data))
    testX, testY = test_data[1], one_hot(list(test_data[0]))
    testX = np.array(testX)


    bs = 128


    vae_mn = VAE(data[0][1].shape, 32, 784)
    vae_mn.train(data, 10, testX/255, testY, bs, test_rate=5)


if __name__=='__main__':
    main()
