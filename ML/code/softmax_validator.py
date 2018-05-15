from __future__ import division
import numpy as np
import utils
import pdb

data = utils.load_dataset("mnist/mnist_test", npy=True)
Xvalid, yvalid = data['X'], data['y']


def kappa(ww, delta):

    ww = np.array(ww)
    yhat = np.sign(np.dot(Xvalid, ww))

    ww2 = np.array(ww + delta)
    yhat2 = np.sign(np.dot(Xvalid, ww2))

    P_A = np.sum(yhat == yhat2) / float(yvalid.size)
    P_E = 0.5

    return (P_A - P_E) / (1 - P_E)


def roni(ww, delta):

    g_err = valid_error(ww)
    new_err = valid_error(ww + delta)

    return new_err - g_err


def valid_error(ww):

    # hardcoded for MNIST
    W = np.reshape(ww, (10, 784))
    # W = np.reshape(ww, (10, 41))

    yhat = np.argmax(np.dot(Xvalid, W.T), axis=1)
    error = np.mean(yhat != yvalid)
    return error


if __name__ == "__main__":
    pdb.set_trace()
