from __future__ import division
import numpy as np
import utils
import pdb

lammy = 0.1
verbose = 1
maxEvals = 10000
X = 0
y = 0
iteration = 1
alpha = 1e-2
d = 0
hist_grad = 0

def init(dataset):

    global X
    X = utils.load_dataset(dataset)['X']

    global y
    y = utils.load_dataset(dataset)['y']

    global d
    d = X.shape[1]

    global hist_grad
    hist_grad = np.zeros(d)

    return d

def funObj(ww, X, y):
    xwy = (X.dot(ww) - y)
    f = 0.5 * xwy.T.dot(xwy)
    g = X.T.dot(xwy)

    return f, g

def funObjL2(ww, X, y):
    xwy = (X.dot(ww) - y)
    f = 0.5 * xwy.T.dot(xwy) + 0.5 * self.lammy * ww.T.dot(ww)
    g = X.T.dot(xwy) + self.lammy * ww

    return f, g

# Reports the direct change to w, based on the given one.
# Batch size could be 1 for SGD, or 0 for full gradient.
def privateFun(theta, ww, batch_size=0):

    global iteration
    print 'python iteration ' + str(iteration) + ' starting'

    ww = np.array(ww)

    # Define constants and params
    nn, dd = X.shape
    threshold = int(d * theta)

    if batch_size > 0 and batch_size < nn:
        idx = np.random.choice(nn, batch_size, replace=False)
    else:
        # Just take the full range
        idx = range(nn)

    f, g = funObj(ww, X[idx, :], y[idx])

    # AdaGrad
    global hist_grad
    hist_grad += g**2

    ada_grad = g / (1e-6 + np.sqrt(hist_grad))

    # Determine the actual step magnitude
    delta = -alpha * ada_grad

    # Weird way to get NON top k values
    if theta < 1:
        param_filter = np.argpartition(
            abs(delta), -threshold)[:d - threshold]
        delta[param_filter] = 0

    w_new = ww + delta
    f_new, g_new = funObj(w_new, X[idx, :], y[idx])

    print 'python iteration ' + str(iteration) + ' ending'
    iteration = iteration + 1

    return delta

def privateFunL2(theta, ww, batch_size=0):

    global iteration
    print 'python iteration ' + str(iteration) + ' starting'

    ww = np.array(ww)

    # Define constants and params
    nn, dd = X.shape
    threshold = int(d * theta)

    if batch_size > 0 and batch_size < nn:
        idx = np.random.choice(nn, batch_size, replace=False)
    else:
        # Just take the full range
        idx = range(nn)

    f, g = funObjL2(ww, X[idx, :], y[idx])

    # AdaGrad
    global hist_grad
    hist_grad += g**2

    ada_grad = g / (1e-6 + np.sqrt(hist_grad))

    # Determine the actual step magnitude
    delta = -alpha * ada_grad

    # Weird way to get NON top k values
    if theta < 1:
        param_filter = np.argpartition(
            abs(delta), -threshold)[:d - threshold]
        delta[param_filter] = 0

    w_new = ww + delta
    f_new, g_new = funObjL2(w_new, X[idx, :], y[idx])

    print 'python iteration ' + str(iteration) + ' ending'
    iteration = iteration + 1

    return delta