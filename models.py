import numpy as np
from time import time
import random
import matplotlib.pyplot as plt


from layers.activations import *
from layers.layers import *
from initializers import *
from utils.data import *
from costs import *
from layers import *
from optim import *
from models import *
from tqdm import tqdm



class Model:
    """

    """
    def __init__(self, input_dim, output_dim, layers=None):
        
        self.layers = layers
        print(self.layers)
        #self.input_dim = np.prod(input_dim) # if first hidden layer fully connected, input is flattened
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.training_loss_history = []
        self.test_loss_history = []
        self.epoch = 0
        self.train_batch_num = 0


    def train_batch(self, x_batch, y_batch):
        self.train_batch_num += 1
        predictions = self.predict(x_batch)
        loss, dx = self.loss(predictions, y_batch)
        self.training_loss_history.append(loss)

        self.calc_gradients(dx)
        self.update_weights()

    def loss(self, predictions, y_batch):
        loss, grads = self.loss_function(y_batch, predictions)
        for layer in self.layers:
            if hasattr(layer, "reg_loss"):
                loss += layer.reg_loss
        return loss, grads

    def update_weights(self):
        for layer in reversed(self.layers):
            if layer._type == "trainable":
                layer.apply_gradients(self.optimizer)

    def calc_gradients(self, dx):
        skip_nb = 0
        for layer in self.layers: # If first layers are not trainable
            if layer._type == "trainable":
                if layer.trainable:
                    break
                else:
                    skip_nb += 1
            else:
                skip_nb += 1
        for layer in list(reversed(self.layers))[:len(self.layers)-skip_nb]:
            dx = layer.compute_gradients(dx)

    def predict(self, x, mode='train'):
        out = x
        for layer in self.layers:
            if layer.name in ['do', 'bn']:
                out = layer.forward(out, mode=mode)
            else:
                out = layer.forward(out)

        return out

    def save_layer(self):
        pass

    def train_epoch(self, data, bs, verbose=True, shuffle=True):
        self.epoch+=1
        print(self.epoch)
        if shuffle:
            random.shuffle(data)
        #st = time()
        it = range(0, len(data)-bs, bs)
        for b in (tqdm(it) if verbose else it):
            x_batch, y_batch = next_batch(data, b, bs)
            y_batch = one_hot(y_batch)
            self.train_batch(x_batch/255, y_batch)
        #print("T:", time()-st)

    def train(self, data, epochs, testX, testY, batch_size=128, test_rate=1):

        for i in range(epochs):
            self.train_epoch(data, batch_size)
            if not self.epoch%test_rate:
                self.test(testX, testY)

    def plot_loss(self):
        if self.training_loss_history:
            plt.plot(range(len(self.training_loss_history)), self.training_loss_history)
            plt.show()



class Classifier(Model):

    def __init__(self, input_dim, output_dim, layers=None):
        if layers is None:
            layers = [
                FC(512, trainable=True),
                ReLU(),
                FC(256, trainable=True),
                ReLU(),
                #FC(128, trainable=False),
                #ReLU(),
                FC(output_dim, trainable=True),
                SoftMax(),
            ]

        #self.optimizer = Momentum()
        #self.optimizer = SGD(0.2)
        #self.optimizer = RMSprop(0.001)
        self.optimizer = Adam(1e-3)
        #self.optimizer = Nesterov()
        self.loss_function = cross_entropy_softmax()

        super().__init__(input_dim, output_dim, layers)

    def test(self, testX, testY):
        preds = self.predict(testX, mode='test')
        loss, grads = self.loss_function(preds, testY)
        self.test_loss_history.append(loss)
        print('acc', sum(np.argmax(preds, axis=1) == np.argmax(testY, axis=1))/ len(testY))


class AE(Model):

    def __init__(self, input_dim, output_dim, layers=None):
        if layers is None:
            layers = [
                FC(256, trainable=True),
                ReLU(),
                FC(64, trainable=True),
                ReLU(),
                FC(256, trainable=True),
                ReLU(),
                #FC(128, trainable=False),
                #ReLU(),
                FC(output_dim, trainable=True),
                sigmoid()
            ]
        self.optimizer = Adam(1e-3)
        #self.optimizer = Nesterov()
        self.loss_function = mse()
        super().__init__(input_dim, output_dim, layers)


    def test(self, testX, testY=None, disp=5):
        preds = self.predict(testX, mode='test')
        loss, grads = self.loss(preds, testX)
        self.test_loss_history.append(loss)
        print(loss)

        n = disp
        canvas_orig = np.empty((28*n , 2*28 * n+1))
        for i in range(n):
            batch_x = testX[i*n:i*n+n]
            g = preds[i*n:i*n+n]

            for j in range(n):
                canvas_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                    batch_x[j].reshape([28, 28])
                canvas_orig[i * 28 :(i + 1) * 28, j * 28 + n*28+1:(j + 1) * 28 + n*28+1] = \
                    g[j].reshape([28, 28])
        canvas_orig[:, n*28:n*28+1] = 1

        print("Original Images")
        plt.figure(figsize=(n*2+1, n))
        plt.imshow(canvas_orig, origin="upper", cmap="gray")
        plt.draw()
        plt.show()

    def train_epoch(self, data, bs):
        self.epoch+=1
        print(self.epoch)

        st = time()
        for b in range(0, len(data)-bs, bs):
            x_batch, y_batch = next_batch(data, b, bs)
            #y_batch = one_hot(y_batch)
            x_batch = x_batch.reshape(len(x_batch), 784)

            self.train_batch(x_batch/255, x_batch/255)
        print("T:", time()-st)


class Ensemble:

    def __init__(self, input_dim, output_dim, input_models, output_model=None):
        
        self.layers = layers
        self.input_models = input_models # [{'model': ..., 'layer': ...:int}]
        
        rep_size = 0
        for model in self.input_models:
            i = 0
            for layer in model['model'].layers:
                if hasattr(layer, 'trainable'):
                    layer.trainable = False
                    if i == model['layer']:
                        rep_size += layer.units
                        break
                    i += 1

        print('rep_size', rep_size)

        if output_model is None:
            self.output_model = Classifier(rep_size , output_dim)
        #self.input_dim = np.prod(input_dim) # if first hidden layer fully connected, input is flattened

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.training_loss_history = []
        self.test_loss_history = []
        self.epoch = 0
        self.train_batch_num = 0

    def pre_process(self, x, mode='test'):
        out_fin = None
        for model in self.input_models:
            out = x
            i = 0
            for layer in model['model'].layers:
                if layer.name in ['do', 'bn']:
                    out = layer.forward(out, mode=mode)
                else:
                    out = layer.forward(out)
                if hasattr(layer, 'trainable'):
                    if model['layer'] == i:
                        break
                    i += 1
            if out_fin is None:
                out_fin = out
            else:
                out_fin = np.append(out_fin, out, axis=1)

        act = ReLU()
        out_fin = act(out_fin) # not ideal
        return out_fin

    def predict(self, x, mode='train'):
        out = self.pre_process(x)
        out = self.output_model.predict(out)
        return out

    def train_epoch(self, data, bs):
        self.epoch+=1
        print(self.epoch)

        st = time()
        for b in range(0, len(data)-bs, bs):
            x_batch, y_batch = next_batch(data, b, bs)
            y_batch = one_hot(y_batch)
            x_batch = self.pre_process(x_batch/255)
            self.output_model.train_batch(x_batch, y_batch)
        print("T:", time()-st)

    def train(self, data, epochs, testX, testY, batch_size=128, test_rate=1):
        for i in range(epochs):
            self.train_epoch(data, batch_size)
            if not i%test_rate:
                self.test(testX, testY)

    def test(self, testX, testY):
        preds = self.predict(testX, mode='test')
        loss, grads = self.output_model.loss_function(preds, testY)
        self.test_loss_history.append(loss)
        print('acc', sum(np.argmax(preds, axis=1) == np.argmax(testY, axis=1))/ len(testY))