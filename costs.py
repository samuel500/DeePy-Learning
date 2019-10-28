import numpy as np
from abc import ABC, abstractmethod

class Cost(ABC):
    def __call__(self, labels, predictions):
        loss = self.compute_loss(labels, predictions)
        grads = self.compute_gradients(labels, predictions)
        return loss, grads

    @abstractmethod
    def compute_loss(self, labels, predictions):
        pass

    @abstractmethod
    def compute_gradients(self, labels, predictions):
        pass


class CrossEntropySoftmax(Cost):

    def compute_loss(self, labels, predictions):
        #np.seterr(all='warn')
        eps = 1e-8
        labels = labels.reshape(predictions.shape)

        loss = np.sum(-np.log(predictions[range(len(predictions)), np.argmax(labels, axis=1)]+eps))
        #print(predictions)
        loss /= len(labels)

        return loss

    def compute_gradients(self, labels, predictions):
        labels = labels.reshape(predictions.shape)

        grads = (predictions - labels)
        grads /= len(labels)
        return grads

class CrossEntropy(Cost):
    def compute_loss(self, labels, predictions):
        #np.seterr(all='warn')
        eps = 1e-8
        labels = labels.reshape(predictions.shape)

        loss = np.sum(-labels * np.log(predictions + eps) - (1 - labels) * np.log(1 - predictions + eps))
        #print(predictions)
        loss /= len(labels)

        return loss

    def compute_gradients(self, labels, predictions):
        eps = 1e-8
        labels = labels.reshape(predictions.shape)


        grads = -labels * (1 / (predictions+eps)) + (1 - labels) * (1 / (1 - (predictions+eps)))

        return grads


class MSE(Cost):

    def compute_loss(self, labels, predictions):
        labels = labels.reshape(predictions.shape)
        loss = (0.5*(labels-predictions)**2).mean()
        return loss

    def compute_gradients(self, labels, predictions):
        labels = labels.reshape(predictions.shape)
        grads = (predictions - labels)
        grads /= len(labels)
        return grads

cross_entropy = CrossEntropy
cross_entropy_softmax = CrossEntropySoftmax
mse = MSE
