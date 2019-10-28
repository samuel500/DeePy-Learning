# DeePy Learning

A deep learning library written exclusively in Python. 

## Notes

-NumPy 1.17.1

-Scipy 1.2.0

-Matplotlib 3.0.2


## Usage

Example:

```
from layers.activations import *
from layers.layers import *
from optim import *
from costs import *


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

