# DeePy Learning

A lightweight deep learning library written in Python. 

## Notes

- Python 3.6

- NumPy 1.17.1

- Scipy 1.2.0

- Matplotlib 3.0.2


## Usage


#### Example 1:

```
from models import *
from layers.activations import *
from layers.layers import *
from optim import *
from costs import *
from utils.data import *


data = load_mnist()
data = list(data)

#test data
test_data = list(load_mnist(dataset='testing'))
test_data = list(zip(*test_data))
testX, testY = test_data[1], one_hot(list(test_data[0]))
testX = np.array(testX)

layers = [
    FC(512),
    ReLU(),
    Batchnorm(),
    FC(32),
    ReLU(),
    FC(10),
    SoftMax()
]


optimizer = Nesterov(learning_rate=2e-2, momentum=0.95)
loss = cross_entropy_softmax()
nn_mnist = Classifier(784, 10, layers=layers, optimizer=optimizer, loss_function=loss)
nn_mnist.train(data, epochs=40, testX=testX/255, testY=testY, batch_size=32, test_rate=50)

```



#### Example 2:

```
from vae import VAE
from utils.data import *

data = load_mnist()
data = list(data)

#test data
test_data = list(load_mnist(dataset='testing'))
test_data = list(zip(*test_data))
testX, testY = test_data[1], one_hot(list(test_data[0]))
testX = np.array(testX)


vae_mn = VAE(data[0][1].shape, latent_dim=32, output_dim=784)
vae_mn.train(data, 10, testX/255, testY, batch_size=128, test_rate=5)

```

![alt text](https://github.com/samuel500/DeePy-Learning/blob/master/vae_10epochs.png)



## Disclaimer

This is primarily an educational tool.
