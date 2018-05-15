from __future__ import division
from numpy.linalg import norm
import numpy as np
import utils
import pdb
import emcee

lammy = 1e-2
X = 0
y = 0
iteration = 1
alpha = 1e-2
d = 0
hist_grad = 0
epsilon = 0

n_classes = 10

scale = False
diffpriv = False


def init(dataset, epsilon):

    passedEpsilon = epsilon
    data = utils.load_dataset(dataset, npy=True)

    global X
    X = data['X']

    global y
    y = data['y']

    # Different for softmax
    global d
    d = X.shape[1] * n_classes

    global hist_grad
    hist_grad = np.zeros(d)

    global samples
    samples = []

    def lnprob(x, alpha):
        return -(alpha / 2) * np.linalg.norm(x)

    if diffpriv:

        nwalkers = max(4 * d, 250)
        sampler = emcee.EnsembleSampler(nwalkers, d, lnprob, args=[passedEpsilon])

        p0 = [np.random.rand(d) for i in range(nwalkers)]
        pos, _, state = sampler.run_mcmc(p0, 100)

        sampler.reset()
        sampler.run_mcmc(pos, 1000, rstate0=state)

        print("Mean acceptance fraction:", np.mean(sampler.acceptance_fraction))

        samples = sampler.flatchain

    return d


def funObj(ww, X, y, batch_size):

    n, d = X.shape

    W = np.reshape(ww, (n_classes, d))

    y_binary = np.zeros((n, n_classes)).astype(bool)
    y_binary[np.arange(n), y.astype(int)] = 1

    XW = np.dot(X, W.T)
    Z = np.sum(np.exp(XW), axis=1)

    # Calculate the function value
    f = - np.sum(XW[y_binary] - np.log(Z))

    # Calculate the gradient value
    res = np.dot((np.exp(XW) / Z[:, None] - y_binary).T, X)

    # Some DP methods only work if the gradient is scaled down to have a max norm of 1
    if scale:
        g = (1 / batch_size) * res / max(1, np.linalg.norm(res)) + lammy * W
    else:
        g = (1 / batch_size) * res + lammy * W

    return f, g.flatten()


# Reports the direct change to w, based on the given one.
# Batch size could be 1 for SGD, or 0 for full gradient.
def privateFun(theta, ww, batch_size=0):

    global iteration
    ww = np.array(ww)

    # Define constants and params
    nn, dd = X.shape

    if batch_size > 0 and batch_size < nn:
        idx = np.random.choice(nn, batch_size, replace=False)
    else:
        # Just take the full range
        idx = range(nn)

    f, g = funObj(ww, X[idx, :], y[idx], batch_size)

    if diffpriv:
        d1, _ = samples.shape
        Z = samples[np.random.randint(0, d1)]
        delta = -alpha * (g + (1 / batch_size) * Z)
    else:
        delta = -alpha * g

    # w_new = ww + delta
    # f_new, g_new = funObj(w_new, X[idx, :], y[idx], batch_size)
    iteration = iteration + 1

    return delta
