import os
import struct
import random
import numpy as np


def next_batch(data, index=0, batch_size=8):
    x_batch = []
    y_batch = []
    for y, x in data[index:index+batch_size]:
        x_batch.append(x)
        y_batch.append(y)
    return np.array(x_batch), np.array(y_batch)


def rand_batch(data, batch_size=32):
    x_batch = []
    y_batch = []
    sample = random.sample(data, k=batch_size)
    for y, x in sample:
        x_batch.append(x)
        y_batch.append(y)
    return np.array(x_batch), np.array(one_hot(y_batch))


def one_hot(labels, max_label = 10):
    return np.eye(max_label)[labels]


def load_mnist(dataset = "training", path = "/home/sam/datasets/MNIST"):
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """

    print("dataset", dataset)
    if dataset == "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError #, "dataset must be 'testing' or 'training'"

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    get_img = lambda idx: (lbl[idx], img[idx])

    # Create an iterator which returns each image in turn
    for i in range(len(lbl)):
        yield get_img(i)

