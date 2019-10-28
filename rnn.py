import numpy as np
from time import time
import random
import matplotlib.pyplot as plt


from layers.activations import *
from layers.layers import *
from layers.abstract_layers import TrainableLayer
from initializers import *
from utils.data import *
from costs import *
from optim import *
from models import *



class RNNLayer(TrainableLayer):

    name = ''
    _type = "trainable"
    dim_mul = 1

    def __init__(self, units, use_bias=True, input_shape=None, kernel_initializer='glorot_uniform', 
                                                               recurrent_initializer='orthogonal', 
                                                               bias_initializer='zeros', 
                                                               trainable=True):
        super().__init__(trainable)
        self.units = units
        self.dim_mul = type(self).dim_mul
        self.input_shape = input_shape
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer #torch_lstm
        self.recurrent_initializer = recurrent_initializer #torch_lstm 
        self.bias_initializer = bias_initializer

        self.activation = Tanh(use_cache=False)
        self.cache = {}
        if input_shape:
            self._initialize()

    def __repr__(self):
        return 'RNNLayer(' + str(self.units) + ')'


    def _initialize(self):
        x_shape = (self.input_shape[1], self.units*self.dim_mul)
        h_shape = (self.units, self.units*self.dim_mul)
        if type(self.kernel_initializer) is str:
            self.kernel_initializer = get_initializer(self.kernel_initializer)
        if type(self.recurrent_initializer) is str:
            self.recurrent_initializer = get_initializer(self.recurrent_initializer)

        self.Wx = self.kernel_initializer(x_shape)
        self.Wh = self.recurrent_initializer(h_shape)

        if self.use_bias:
            if type(self.bias_initializer) is str:
                self.bias_initializer = get_initializer(self.bias_initializer)
            self.B = self.bias_initializer((self.units*self.dim_mul))
        else:
            self.B = None
        

    def forward(self, x):

        N, T, D = x.shape

        if not self.input_shape:
            self.input_shape = x.shape[1:]
            self._initialize()

        H = self.units
        prev_h = np.zeros(H)
        h = np.zeros((N, T+1, H))
        h[:,0,:] = prev_h
        self.cache_x = x
        for i in range(T):
            next_h = x[:,i,:].dot(self.Wx) + prev_h.dot(self.Wh)
            if self.use_bias:
                next_h += self.B
            next_h = self.activation(next_h)
            h[:,i+1,:] = next_h
            prev_h = next_h
        self.cache_h = h
        return h[:,-1,:]


    def compute_gradients(self, dh):

        N, T, D = self.cache_x.shape

        prev_h = self.cache_h[:,-2,:]
        next_h = self.cache_h[:,-1,:]
        x = self.cache_x

        dnext_tanh = self.activation.compute_gradients(d=dh, cache=next_h)
        dxl = dnext_tanh.dot(self.Wx.T)                       
        dprev_h = dnext_tanh.dot(self.Wh.T)
        dWx = (x[:,-1,:].T).dot(dnext_tanh)
        dWh = (prev_h.T).dot(dnext_tanh)
        db = None
        if self.use_bias:
            db = dnext_tanh.sum(axis=0) 

        dx = np.zeros_like(self.cache_x)
        dx[:,T-1,:] = dxl

        for i in range(T-1, 0, -1):
            prev_h = self.cache_h[:,i-1,:]
            next_h = self.cache_h[:,i,:]

            dnext_tanh = self.activation.compute_gradients(d=dprev_h, cache=next_h)
                      
            dxl = dnext_tanh.dot(self.Wx.T)                         
            dx[:,i-1,:] = dxl

            dprev_h = dnext_tanh.dot(self.Wh.T)                  
            dWx += (x[:,i-1,:].T).dot(dnext_tanh)                       
            dWh += (prev_h.T).dot(dnext_tanh) 
            if self.use_bias:              
                db += dnext_tanh.sum(axis=0)

        dh0 = dprev_h

        self.gradients = {'Wx': dWx, 'Wh': dWh, 'B': db}
        # dx?

        return dh0

    def apply_gradients(self, optimizer, **kwargs):
        if self.trainable:
            if self.use_bias:
                optimizer(self.B, self.gradients['B'], **kwargs)
            optimizer(self.Wx, self.gradients['Wx'], **kwargs)
            optimizer(self.Wh, self.gradients['Wh'], **kwargs)



class LSTMLayer(RNNLayer):

    dim_mul = 4

    def __repr__(self):
        return 'LSTMLayer(' + str(self.units) + ')'


    def _initialize(self, unit_forget_bias=True):
        super()._initialize()
        if unit_forget_bias: # http://proceedings.mlr.press/v37/jozefowicz15.pdf
            H = self.units
            if self.use_bias:
                self.B[H:H*2] = 1
        self.recurrent_activation = HardSigmoid(use_cache=False)
        #self.recurrent_activation = Sigmoid(use_cache=False)


    def forward(self, x):
        
        N, T, D = x.shape
        if not self.input_shape:
            self.input_shape = x.shape[1:]
            self._initialize()

        H = self.units

        prev_h = np.zeros((N, H))
        prev_c = np.zeros(H)
        h = np.zeros((N, T+1, H))
        c = np.zeros((N, T+1, H))
        h[:,0,:] = prev_h
        c[:,0,:] = prev_c

    
        cache_tanhc = []
        cache_ifog = []
        self.cache = {}
        self.cache['x'] = x
        for y in range(T):
            ifog = x[:,y,:].dot(self.Wx) + prev_h.dot(self.Wh) 
            if self.use_bias:
                ifog += self.B
            ifog[:,0:3*H] = self.recurrent_activation(ifog[:,0:3*H]) # input, forget and output gates                 
            ifog[:,3*H:4*H] = self.activation(ifog[:,3*H:4*H]) # Gate gate 
            cache_ifog.append(ifog)
            i, f, o, g = ifog[:,:H], ifog[:,H:2*H], ifog[:,2*H:3*H], ifog[:,3*H:4*H]

            next_c = f * prev_c + i * g                       
            next_c_tanh = self.activation(next_c) 
            cache_tanhc.append(next_c_tanh)                       
            next_h = o * next_c_tanh                             
            h[:,y+1,:] = next_h
            c[:,y+1,:] = next_c
            prev_h, prev_c = next_h, next_c
      
        self.cache['ifog'] = cache_ifog
        self.cache['tanhc'] = cache_tanhc
        self.cache['h'] = h
        self.cache['c'] = c

        return h[:,-1,:]


    def compute_gradients(self, dh):

        N, T, D = self.cache['x'].shape

        H = self.units 

        x = self.cache['x']

        dnext_c = np.zeros((N,H))
        dnext_h = np.zeros((N,H))
        dx = np.zeros((N, T, D))
        dWx = np.zeros((D, 4*H))
        dWh = np.zeros((H, 4*H))

        db = None
        if self.use_bias:
            db = np.zeros(4*H)

        dnext_h = dh

        for y in range(T, 0, -1):
            prev_h = self.cache['h'][:,y-1,:]
            prev_c = self.cache['c'][:,y-1,:]
            next_c_tanh = self.cache['tanhc'][y-1]
            ifog = self.cache['ifog'][y-1]
            i, f, o, g = ifog[:,:H], ifog[:,H:2*H], ifog[:,2*H:3*H], ifog[:,3*H:4*H]
            dgate = ifog.copy()

            # (Hard) Sigmoid gradient
            dgate[:,:3*H] = self.recurrent_activation.compute_gradients(cache=dgate[:,:3*H])

            # TanH gradient
            dgate[:,3*H:4*H] = self.activation.compute_gradients(cache=dgate[:,3*H:4*H])
            dnc_tanh = self.activation.compute_gradients(cache=next_c_tanh)


            dnc_prod = dnext_h * o * dnc_tanh + dnext_c
            dgate[:,:H] *= dnc_prod * g

            dgate[:,H:2*H] *= dnc_prod * prev_c
            dgate[:,2*H:3*H] *= dnext_h * next_c_tanh
            dgate[:,3*H:4*H] *= dnc_prod * i

            # calculate final gradients
            dx[:,y-1,:] = dgate.dot(self.Wx.T)
            dprev_h = dgate.dot(self.Wh.T)
            dprev_c = dnext_c * f + dnext_h * o * dnc_tanh * f
            dWx += x[:,y-1,:].T.dot(dgate) 
            dWh += prev_h.T.dot(dgate)
            if self.use_bias:
                db += dgate.sum(axis=0)

            
            dnext_h = dprev_h
            dnext_c = dprev_c 
        dh0 = dprev_h

        self.gradients = {'Wx': dWx, 'Wh': dWh, 'B': db}
        # dx?

        return dh0 #? 



class GRULayer(RNNLayer):

    dim_mul = 3

    def __repr__(self):
        return 'GRULayer(' + str(self.units) + ')'


    def _initialize(self):
        super()._initialize()
        self.recurrent_activation = HardSigmoid(use_cache=False)
        #self.recurrent_activation = Sigmoid(use_cache=False)

    def forward(self, x):
        N, T, D = x.shape
        if not self.input_shape:
            self.input_shape = x.shape[1:]
            self._initialize()

        H = self.units

        prev_h = np.zeros((N, H))
        h = np.zeros((N, T+1, H))
        h[:,0,:] = prev_h

        self.cache['x'] = x
        cache_rzh = []
        for y in range(T):
            rzh = x[:,y,:].dot(self.Wx) 
            if self.use_bias:
                rzh += self.B
            rzh[:,:2*H] += prev_h.dot(self.Wh[:,:2*H]) 
            rzh[:,:2*H] = self.recurrent_activation(rzh[:,:2*H])

            rzh[:,2*H:] += rzh[:,:H]*prev_h.dot(self.Wh[:,2*H:])
            rzh[:,2*H:] = self.activation(rzh[:,2*H:])

            cache_rzh.append(rzh)
            
            next_h = rzh[:,H:2*H]*rzh[:,2*H:] + (1-rzh[:,H:2*H])*prev_h

            h[:,y+1,:] = next_h

            prev_h = next_h

        self.cache['rzh'] = cache_rzh
        self.cache['h'] = h

        return next_h #h[:,-1,:]


    def compute_gradients(self, dh):
        N, T, D = self.cache['x'].shape
        H = self.units 

        x = self.cache['x']


        dnext_h = np.zeros((N,H))
        dx = np.zeros((N, T, D))
        dWx = np.zeros((D, 3*H))
        dWh = np.zeros((H, 3*H))

        db = None
        if self.use_bias:
            db = np.zeros(3*H)

        dnext_h = dh

        for y in range(T, 0, -1):
            prev_h = self.cache['h'][:,y-1,:]
            rzh = self.cache['rzh'][y-1]
            r, z, h = rzh[:,:H], rzh[:,H:2*H], rzh[:,2*H:]
            dgate = rzh.copy()

            dgate[:,:2*H] = self.recurrent_activation.compute_gradients(d=1, cache=dgate[:,:2*H])
            dgate[:,2*H:] = self.activation.compute_gradients(d=1, cache=dgate[:,2*H:])

            dgate[:,H:2*H] = dnext_h*(h-prev_h)*dgate[:,H:2*H] #dz
            dgate[:,2*H:] = dnext_h*z*dgate[:,2*H:] #dht
            dgate[:,:H] = dgate[:,2*H:]*(prev_h.dot(self.Wh[:,2*H:]))*dgate[:,:H] #dr

            # calculate final gradients
            dx[:,y-1,:] = dgate.dot(self.Wx.T)
            dprev_h = (1-z)*dnext_h #

            dprev_h += dgate[:,:2*H].dot(self.Wh[:,:2*H].T)
            dprev_h += r*dgate[:,2*H:].dot(self.Wh[:,2*H:].T)

            dWx += x[:,y-1,:].T.dot(dgate) 
            dWh[:,:2*H] += prev_h.T.dot(dgate[:,:2*H])

            dWh[:,2*H:] += (r*prev_h).T.dot(dgate[:,2*H:])

            if self.use_bias:
                db += dgate.sum(axis=0)

            dnext_h = dprev_h

        dh0 = dprev_h

        self.gradients = {'Wx': dWx, 'Wh': dWh, 'B': db}

        return dh0 