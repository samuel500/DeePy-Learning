import numpy as np
from time import time
import random
import matplotlib.pyplot as plt


from layers.activations import *
from layers.layers import *
from initializers import *
from utils.data import *
from costs import *
from optim import *
from models import *
from scipy.stats import norm

class VAE:
    """

    """
    def __init__(self, input_dim, latent_dim, output_dim, layers=None):

        self.layers = layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.training_loss_history = []
        self.test_loss_history = []
        self.epoch = 0
        self.train_batch_num = 0
        self.encoder = self.VAEEncoder(input_dim, latent_dim)
        self.decoder = self.VAEDecoder(latent_dim, output_dim)

        self.optimizer = Adam(1e-3)

    def train_batch(self, x_batch, y_batch):
        self.train_batch_num += 1
        predictions, mu, logvar = self.predict(x_batch)
        loss, dx, dmu, dlogvar = self.loss(predictions, y_batch, mu, logvar)
        self.training_loss_history.append(loss)

        dx = self.decoder.calc_gradients(dx)

        self.encoder.calc_gradients(dx+dmu, dx+dlogvar)

        self.decoder.update_weights(self.optimizer)
        self.encoder.update_weights(self.optimizer)


    def loss(self, predictions, y_batch, mu, logvar):
        encoder_loss, mu_grads, logvar_grads = self.encoder.loss(mu, logvar)
        decoder_loss, grads = self.decoder.loss(predictions, y_batch)
        loss = encoder_loss + decoder_loss

        return loss, grads, mu_grads, logvar_grads


    def predict(self, x, mode='train'):
        mu, logvar = self.encoder.forward(x)

        sample_z = mu + np.exp(logvar * .5) * np.random.standard_normal(size=(len(x), self.latent_dim))

        decode = self.decoder.forward(sample_z)

        return decode, mu, logvar


    def save_layer(self):
        pass

    def train_epoch(self, data, bs):
        self.epoch+=1
        print(self.epoch)

        st = time()
        for b in range(0, len(data)-bs, bs):
            x_batch, y_batch = next_batch(data, b, bs)
            x_batch = x_batch.reshape(len(x_batch), 784)

            self.train_batch(x_batch/255, x_batch/255)
        print("T:", time()-st)


    def train(self, data, epochs, testX, testY, batch_size=128, test_rate=1):
        self.batch_size = batch_size
        for i in range(epochs):
            self.train_epoch(data, batch_size)
            if not self.epoch%test_rate:
                self.test(testX, testY)


    def plot_loss(self):
        if self.training_loss_history:
            plt.plot(range(len(self.training_loss_history)), self.training_loss_history)
            plt.show()


    def test(self, testX, testY, disp=5):
        preds, mu, logvar = self.predict(testX, mode='test')
        loss, grads, _, _ = self.loss(preds, testX, mu, logvar)
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


        if self.latent_dim == 2:
            self.latent2d()

            self.latent2d2(testX, testY)


    def generate_img(self, n=5):


        z = np.random.standard_normal(size=(n, self.latent_dim))

        preds = self.decoder.forward(z)

        canvas_orig = np.empty((28, 28*n))

        i=0
        g = preds

        for j in range(n):

            canvas_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                g[j].reshape([28, 28])

        print("Original Images")
        plt.figure(figsize=(n*2+1, n))
        plt.imshow(canvas_orig, origin="upper", cmap="gray")
        plt.draw()
        plt.show()

    def latent2d(self):

        n = 15
        digit_size = 28
        figure = np.zeros((digit_size*n, digit_size*n))

        grid_x=norm.ppf(np.linspace(0.05,0.95,n))
        grid_y=norm.ppf(np.linspace(0.05,0.95,n))

        for i, yi in enumerate(grid_x):
            for j,xi in enumerate(grid_y):
                z_sample=np.array([[xi,yi]])
                x_decoded=self.decoder.forward(z_sample)
                digit=x_decoded[0].reshape(digit_size,digit_size)
                figure[i*digit_size:(i+1)*digit_size,
                      j*digit_size:(j+1)*digit_size]=digit

        plt.figure(figsize=(10,10))
        plt.imshow(figure, cmap='Greys_r')
        plt.show()
        #encoder=Model(X, z_mean)

        # a 2d plot of 10 digit classes in latent space
        #x_test_encoded=self.encoder.forward(x_test, batch_size=batch_size)
    def latent2d2(self, X, Y):
        s = len(Y)
        mu, logvar = self.encoder.forward(X[:s])
        x_test_encoded = mu + np.exp(logvar * 0.5) * np.random.standard_normal(size=(s, self.latent_dim))

        plt.figure(figsize=(6,6))
        #print(Y)
        Y = np.argmax(Y, axis=1)
        #Y=Y[:s]
        #print(Y)
        plt.scatter(x_test_encoded[:,0], x_test_encoded[:,1], c=Y)
        plt.colorbar()
        plt.show()



    class VAEEncoder:
        def __init__(self, input_dim, latent_dim, layers=None):
            if layers is None:
                self.layers = [
                    Batchnorm(trainable=True),
                    #Layernorm(trainable=False),
                    FC(256),
                    ReLU(),
                ]
            else:
                self.layers = layers
            self.mean_layer = FC(latent_dim, trainable=True)
            self.logvar_layer = FC(latent_dim, trainable=True)
            #self.optimizer = Nesterov()

        def forward(self, x, mode='train'):
            out = x
            for layer in self.layers:
                if layer.name in ['do', 'bn']:
                    out = layer.forward(out, mode=mode)
                else:
                    out = layer.forward(out)
            mu = self.mean_layer(out)
            logvar = self.logvar_layer(out)
            return mu, logvar


        def calc_gradients(self, dmu, dlogvar):
            skip_nb = 0
            dx1 = self.mean_layer.compute_gradients(dmu)
            dx2 = self.logvar_layer.compute_gradients(dlogvar)
            dx = dx1 + dx2
            for layer in self.layers: # If first layers are not trainable, no need to calc gradients
                if layer._type == "trainable":
                    if layer.trainable:
                        break
                    else:
                        skip_nb += 1
                else:
                    skip_nb += 1
            for layer in list(reversed(self.layers))[:len(self.layers)-skip_nb]:
                dx = layer.compute_gradients(dx)


        def update_weights(self, optimizer):
            for layer in reversed(self.layers):
                if layer._type == "trainable":
                    layer.apply_gradients(optimizer)


        def loss(self, mu, logvar):
            beta = 1
            kl = -0.5 * np.sum(1 + logvar - np.power(mu, 2) - np.exp(logvar))
            kl *=beta
            mu_grads = mu
            mu_grads *= beta
            logvar_grads = -0.5 * (1 - np.exp(logvar))
            logvar_grads *= beta
            return kl, mu_grads, logvar_grads



    class VAEDecoder:
        def __init__(self, latent_dim, output_dim, layers=None):
            if layers is None:
                self.layers = [
                    FC(256),
                    #Batchnorm(trainable=True),

                    ReLU(),
                    FC(output_dim),
                    #Batchnorm(),
                    Sigmoid()
                ]
            else:
                self.layers = layers
            self.loss_function = cross_entropy()
            #self.loss_function = mse()

        def forward(self, x, mode='train'):
            out = x
            for layer in self.layers:
                if layer.name in ['do', 'bn']:
                    out = layer.forward(out, mode=mode)
                else:
                    out = layer.forward(out)

            return out


        def calc_gradients(self, dx):
            for layer in list(reversed(self.layers)):
                dx = layer.compute_gradients(dx)
            return dx

        def update_weights(self, optimizer):
            for layer in reversed(self.layers):
                if layer._type == "trainable":
                    layer.apply_gradients(optimizer)

        def loss(self, predictions, y_batch):
            loss, grads = self.loss_function(y_batch, predictions)
            for layer in self.layers:
                if hasattr(layer, "reg_loss"):
                    loss += layer.reg_loss
            return loss, grads
