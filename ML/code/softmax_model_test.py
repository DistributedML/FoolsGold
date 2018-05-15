from __future__ import division
import numpy as np
import utils
import matplotlib.pyplot as plt
import pdb

# #datatest = utils.load_dataset("mnist_test", npy=True)
# datatest = utils.load_dataset("mnist/mnist_test", npy=True)
# Xtest, ytest = datatest['X'], datatest['y']
#
# #datatrain = utils.load_dataset("mnist_train", npy=True)
# datatrain = utils.load_dataset("mnist/mnist_train", npy=True)
# XBin, yBin = datatrain['X'], datatrain['y']

class SoftMaxModelTest:
    def __init__(self, dataset, numClasses, numFeatures):
        self.datatest = utils.load_dataset(dataset+'/'+dataset+"_test", npy=True)
        self.datatrain = utils.load_dataset(dataset+'/'+dataset+"_train", npy=True)
        self.Xtest, self.ytest = self.datatest['X'], self.datatest['y']
        self.XBin, self.yBin = self.datatrain['X'], self.datatrain['y']
        self.num_classes = numClasses
        self.num_features = numFeatures

    def train_error(self, ww):

        # hardcoded for MNIST
        W = np.reshape(ww, (self.num_classes, self.num_features))
        #W = np.reshape(ww, (10, 41))
        yhat = np.argmax(np.dot(self.XBin, W.T), axis=1)
        error = np.mean(yhat != self.yBin)
        return error


    def test_error(self, ww):
        W = np.reshape(ww, (self.num_classes, self.num_features))
        #W = np.reshape(ww, (23, 41))
        yhat = np.argmax(np.dot(self.Xtest, W.T), axis=1)
        error = np.mean(yhat != self.ytest)
        return error


    def kappa(self, ww, delta):

        ww = np.array(ww)
        yhat = np.argmax(np.dot(self.Xtest, ww), axis=1)

        ww2 = np.array(ww + delta)
        yhat2 = np.argmax(np.dot(self.Xtest, ww2), axis=1)

        P_A = np.mean(yhat == yhat2)
        P_E = 0.5

        return (P_A - P_E) / (1 - P_E)


    def roni(self, ww, delta):

        ww = np.array(ww)
        yhat = np.argmax(np.dot(self.Xtest, ww), axis=1)

        ww2 = np.array(ww + delta)
        yhat2 = np.argmax(np.dot(self.Xtest, ww2), axis=1)

        g_err = np.mean(yhat != self.ytest)
        new_err = np.mean(yhat2 != self.ytest)

        # How much does delta improve the validation error?
        return g_err - new_err


    def plot(self, data):

        data = np.loadtxt("lossflush.csv", delimiter=',')
        fig = plt.figure()
        plt.plot(data)
        fig.savefig("loss.jpeg")

        return 1
