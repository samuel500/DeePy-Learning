import numpy as np
from layers.activations import *
from layers.abstract_layers import TrainableLayer, Layer
from initializers import *


class FC(TrainableLayer):

    def __init__(self, units, input_shape=None, use_bias=True, weight_initializer='default', bias_initializer='zeros', l1_reg=0, l2_reg=0, trainable=True):
        super().__init__(trainable)
        self.units = units
        self.input_shape = input_shape
        self.use_bias = use_bias
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer

        self.feedback_al = False
        if input_shape:
            self._initialize()
            if self.feedback_al:
                self.fa_W = self.weight_initializer((np.prod(self.input_shape), self.units)) # feedback alignment test, Lilicrap et al (2016)

        self.trainable = trainable

        self.l1_reg = l1_reg
        self.l2_reg = l2_reg

    def __repr__(self):
        return "FC(" + str(self.units) + ")"

    def _initialize(self):
        shape = (np.prod(self.input_shape), self.units)
        if type(self.weight_initializer) is str:
            self.weight_initializer = get_initializer(self.weight_initializer)

            self.W = self.weight_initializer(shape)

        if self.use_bias:
            if type(self.bias_initializer) is str:
                self.bias_initializer = get_initializer(self.bias_initializer)
            self.B = self.bias_initializer((self.units))
        else:
            self.B = None


    def forward(self, x):
        if not self.input_shape:
            self.input_shape = x.shape[1:]
            #print(x.shape)
            self._initialize()
            if self.feedback_al:
                self.fa_W = self.weight_initializer((np.prod(self.input_shape), self.units))
        x = x.reshape(x.shape[0], np.prod(x.shape[1:])) # flatten input
        out = x.dot(self.W)
        if self.use_bias:
            out += self.B
        self.cache = x
        return out


    def compute_gradients(self, d):
        x = self.cache

        if self.feedback_al:
            dx = d.dot(self.fa_W.T)
        else:
            dx = d.dot(self.W.T)
        dx = dx.reshape(x.shape)
        dw = x.T.dot(d)
        if self.use_bias:
            db = d.sum(axis=0)
        else:
            db = None

        dw += self.l2_reg * 2 * self.W + self.l1_reg * np.sign(self.W)
        self.gradients = {'W': dw, 'B': db}

        return dx


    def apply_gradients(self, optimizer, **kwargs):
        if self.trainable:
            if self.use_bias:
                optimizer(self.B, self.gradients['B'], **kwargs)
            optimizer(self.W, self.gradients['W'], **kwargs)


    @property
    def reg_loss(self):
        loss = 0
        if self.l2_reg:
            loss += self.l2_reg*n.sum(self.W**2)
        if self.l1_reg:
            loss += self.l1_reg*n.sum(np.abs(self.W))
        return loss


class Dense(TrainableLayer):

    def __init__(self, units, input_shape=None, activation=None, use_bias=True, weight_initializer='default', bias_initializer='zeros', l1_reg=0, l2_reg=0, trainable=True):
        super().__init__(trainable)
        self.fc = FC(units, input_shape, use_bias, weight_initializer, bias_initializer, l1_reg, l2_reg)
        self.activation = None
        if activation:
            self.activation = activation()

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        out = self.fc(x)
        if self.activation:
            out = self.activation(out)
        return out

    def compute_gradients(self, d):
        if self.activation:
            d = self.activation.compute_gradients(d)
        dx = self.fc.compute_gradients(d)
        return dx

    def apply_gradients(self, optimizer, **kwargs):
        if self.trainable:
            self.fc.apply_gradients(optimizer, **kwargs)

    @property
    def reg_loss(self):
        return self.fc.reg_loss


class Batchnorm(TrainableLayer):

    name = 'bn'
    axis = 0
    def __init__(self, momentum=0.96, epsilon=1e-5, trainable=True, input_shape=None):
        super().__init__(trainable)
        self.momentum = momentum
        self.epsilon = epsilon
        self.running_mean = None
        self.running_var = None
        self.gamma = None
        self.beta = None
        if input_shape is not None:
            self._initialize(input_shape)

    def _initialize(self, shape):
        if not type(self).axis:
            self.running_mean = np.zeros(shape)
            self.running_var = np.ones(shape)
            self.gamma = np.ones(shape)
            self.beta = np.zeros(shape)
        else:
            self.gamma = np.ones(shape).reshape(-1,1)
            self.beta = np.zeros(shape).reshape(-1,1)

    def forward(self, x, mode):
        x = x.reshape(x.shape[0], np.prod(x.shape[1:])) # flatten
        if self.gamma is None and not type(self).axis:
            self._initialize(x.shape[1:])
        elif self.gamma is None and type(self).axis:
            self._initialize(x.T.shape[1:])

        if mode == 'train':
            mean = x.mean(axis=0)
            var = x.var(axis=0) + self.epsilon
            std = np.sqrt(var)
            z = (x - mean)/std
            out = self.gamma * z + self.beta
            # running weighted average
            if not type(self).axis:
                self.running_mean = self.momentum*self.running_mean + (1-self.momentum)*mean
                self.running_var = self.momentum*self.running_var + (1-self.momentum)*(std**2)

            # save values for backward call

            self.cache = {'x':x,'mean':mean,'std':std,'z':z,'var':var}

        elif mode == 'test':
            out = self.gamma * (x - self.running_mean) / np.sqrt(self.running_var + self.epsilon) + self.beta


        return out


    def compute_gradients(self, d):
        ax = type(self).axis
        self.dbeta = d.sum(axis=ax)

        self.dgamma = np.sum(d * self.cache['z'], axis=ax)

        N = d.shape[0]
        z = self.cache['z']
        dfdz = d * self.gamma                                 
        dfdz_sum = np.sum(dfdz,axis=0) 
        dx = dfdz - dfdz_sum/N - np.sum(dfdz * z,axis=0) * z/N
        dx /= self.cache['std']
        return dx


    def apply_gradients(self, optimizer, **kwargs):
        if self.trainable:
            optimizer(self.gamma, self.dgamma)
            optimizer(self.beta, self.dbeta)

'''
class Layernorm(Batchnorm):
    name = 'ln'
    axis = 1

    def forward(self, x, mode='train'):
        out = super().forward(x.T, mode)
        return out.T

    def compute_gradients(self, d):
        dx = super().compute_gradients(d.T)
        self.dbeta = self.dbeta.reshape(-1,1)
        self.dgamma = self.dgamma.reshape(-1,1)
        return dx.T
'''

class Dropout(Layer):
    name = 'do'
    def __init__(self, p, refresh_mask=1, grads_only=False):
        self.p = p # Probability of keeping an activation
        self.grads_only = grads_only
        self.it = 0
        self.refresh_mask = refresh_mask
        self.mask = None

    def __repr__(self):
        r = "Dropout(p="+str(self.p)+", refresh_mask="+str(self.refresh_mask)+", grads_only="+str(self.grads_only)+"; it: "+str(self.it)
        return r

    def forward(self, x, mode):
        self.it += 1
        self.mode = mode
        if not self.grads_only:
            if mode == 'train':
                if self.mask is None or not self.it % self.refresh_mask:
                    self.mask = (np.random.rand(*x.shape) < self.p) / self.p
                out = x * self.mask
                self.cache = self.mask
            elif mode == 'test':
                out = x
            else:
                raise
        else:
            out = x
            if mode == 'train':
                if self.mask is None or not self.it % self.refresh_mask:
                    self.mask = (np.random.rand(*x.shape) < self.p) / self.p
                # out = x * self.mask
                self.cache = self.mask
            elif mode == 'test':
                pass
            else:
                raise

        return out

    def compute_gradients(self, d):
        if self.mode == 'train':
            dx = d * self.cache
        elif self.mode == 'test':
            dx = d
        return dx


#class




class Layernorm(TrainableLayer):
    name = 'ln'
    axis = 1
    def __init__(self, momentum=0.96, epsilon=1e-5, trainable=True, input_shape=None):
        super().__init__(trainable)
        self.momentum = momentum
        self.epsilon = epsilon
        self.gamma = None
        self.beta = None
        if input_shape is not None:
            self._initialize(input_shape)

    def _initialize(self, shape):
        self.gamma = np.ones(shape)
        self.beta = np.zeros(shape)

    def forward(self, x, mode='train'):
        x = x.reshape(x.shape[0], np.prod(x.shape[1:])) # flatten input

        if self.gamma is None and type(self).axis:
            print(x.shape)
            self._initialize(x.shape[1:])

        mean = x.mean(axis=1)
        var = x.var(axis=1) + self.epsilon
        std = np.sqrt(var)
        z = (x.T - mean)
        z /= std
        z=z.T
        out = self.gamma * z + self.beta
        #out = out.T

        # save values for backward call

        self.cache = {'x':x,'mean':mean,'std':std,'z':z,'var':var}



        return out


    def compute_gradients(self, d):
        ax = type(self).axis
        self.dbeta = d.sum(axis=0)
        self.dgamma = np.sum(d * self.cache['z'], axis=0)
        #print('db', self.dbeta.shape)

        N = d.shape[0]
        z = self.cache['z']
        dfdz = d * self.gamma                     
        dfdz_sum = np.sum(dfdz,axis=0)                           
        dx = dfdz - dfdz_sum/N - np.sum(dfdz * z,axis=0) * z/N          

        #print('std', self.cache['std'].shape)

        #print('dx', dx.shape)
        dx = dx.T
        dx /= self.cache['std']
        dx = dx.T

        return dx


    def apply_gradients(self, optimizer, **kwargs):
        #print('gamma', self.gamma.shape)
        #print('dgamma', self.dgamma.shape)
        if self.trainable:
            optimizer(self.gamma, self.dgamma)
            optimizer(self.beta, self.dbeta)

#'''
