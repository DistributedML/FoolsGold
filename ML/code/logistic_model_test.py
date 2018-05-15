from __future__ import division
import numpy as np
import utils
import pdb

data = utils.load_dataset("credittest")
Xtest, ytest = data['X'], data['y']

data = utils.load_dataset("credittrain")
XBin, yBin = data['X'], data['y']


def train_error(ww):
    ww = np.array(ww)
    yhat = np.sign(np.dot(XBin, ww))
    error = np.sum(yhat != yBin) / float(yBin.size)
    return error


def test_error(ww):
    ww = np.array(ww)
    yhat = np.sign(np.dot(Xtest, ww))
    error = np.sum(yhat != ytest) / float(yhat.size)
    return error

def kappa(ww, delta):
    
    ww = np.array(ww)
    yhat = np.sign(np.dot(Xtest, ww))
    
    ww2 = np.array(ww + delta)
    yhat2 = np.sign(np.dot(Xtest, ww2))

    P_A = np.sum(yhat == yhat2) / float(ytest.size)
    P_E = 0.5

    return (P_A - P_E) / (1 - P_E)

def roni(ww, delta):
    
    ww = np.array(ww)
    yhat = np.sign(np.dot(Xtest, ww))
    
    ww2 = np.array(ww + delta)
    yhat2 = np.sign(np.dot(Xtest, ww2))

    g_err = np.sum(yhat != ytest) / float(ytest.size)
    new_err = np.sum(yhat2 != ytest) / float(ytest.size)

    # How much does delta improve the validation error?
    return g_err - new_err

def plot(data):

    data = np.loadtxt("lossflush.csv", delimiter=',')
    fig = plt.figure()
    plt.plot(data)
    fig.savefig("loss.jpeg")

    return 1