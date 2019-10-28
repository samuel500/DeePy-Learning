import numpy as np
from layers.activations import *


def get_initializer(name: str, activation=None):
    if not activation and name == 'default':
        return variance_scaling_initializer
    elif name in {'xavier', 'xavier_init', 'glorot'}:
        return xavier_init
    elif name in {'he', 'he_init'}:
        return he_init
    elif name in {'glorot_uniform'}:
        return glorot_uniform
    elif activation == relu and name == 'default':
        return he_init
    elif name in {'orthogonal'}:
        return orthogonal
    elif hasattr(np, name):
        return getattr(np, name)
    else:
        return xavier_init


def variance_scaling_initializer(shape: tuple, scale=2, mode = 'FAN_IN', distribution='normal'):
    if mode not in ['FAN_IN', 'FAN_OUT', 'FAN_AVG']:
        raise TypeError

    if mode == 'FAN_IN':
        FAN = shape[0]
    elif mode == 'FAN_OUT':
        FAN = shape[1]
    elif mode == 'FAN_AVG':
        FAN = sum(shape)/len(shape)
    else:
        print('mode not recognized')
        raise

    if distribution == 'normal':
        return np.random.normal(size=shape, scale=np.sqrt(scale/FAN))
    elif distribution == 'uniform':
        limit = np.sqrt(3*scale/FAN)
        return np.random.uniform(size=shape, low=-limit, high=limit)
    else:
        print('distribution not recognized')
        raise


def he_init(shape):
    return variance_scaling_initializer(shape, scale=2, mode='FAN_IN')

def xavier_init(shape):
    return variance_scaling_initializer(shape, scale=1, mode='FAN_AVG')

def orthogonal(shape, gain=1.0): 
    """
    https://arxiv.org/pdf/1312.6120.pdf
    """
    A = np.random.normal(size=shape, scale=gain)
    if shape[0] < shape[1]:
        A = A.T
    Q, R =  np.linalg.qr(A)
    if shape[0] < shape[1]:
        Q = Q.T
    return Q

def glorot_uniform(shape):
    return variance_scaling_initializer(shape, scale=2, mode='FAN_AVG', distribution='uniform')

def torch_lstm(shape):
    return variance_scaling_initializer(shape, scale=1/3, mode = 'FAN_OUT', distribution='uniform')
