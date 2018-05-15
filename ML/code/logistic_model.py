from __future__ import division
from numpy.linalg import norm
import numpy as np
import utils
import pdb
import emcee

lammy = 0.01
verbose = 1
X = 0
y = 0
iteration = 1
alpha = 1e-2
d = 0
hist_grad = 0
epsilon = 0

scale = True
diffpriv = False


def init(dataset, epsilon):

    passedEpsilon = epsilon
    data = utils.load_dataset(dataset)

    global X
    X = data['X']

    global y
    y = data['y']

    global d
    d = X.shape[1]

    global hist_grad
    hist_grad = np.zeros(d)

    global samples
    samples = []

    def lnprob(x, alpha):
        return -(alpha / 2) * np.linalg.norm(x)

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
    yXw = y * X.dot(ww)

    # Calculate the function value
    f = np.sum(np.logaddexp(0, -yXw)) + 0.5 * lammy * ww.T.dot(ww)

    # Calculate the gradient value
    res = - y / np.exp(np.logaddexp(0, yXw))
    if scale:
        g = (1 / batch_size) * X.T.dot(res) / \
            max(1, np.linalg.norm(X.T.dot(res))) + lammy * ww
    else:
        g = (1 / batch_size) * X.T.dot(res) + lammy * ww

    return f, g


# Reports the direct change to w, based on the given one.
# Batch size could be 1 for SGD, or 0 for full gradient.
def privateFun(theta, ww, batch_size=0):

    global iteration
    ww = np.array(ww)

    # Define constants and params
    nn, dd = X.shape
    threshold = int(d * theta)

    if batch_size > 0 and batch_size < nn:
        idx = np.random.choice(nn, batch_size, replace=False)
    else:
        # Just take the full range
        idx = range(nn)

    f, g = funObj(ww, X[idx, :], y[idx], batch_size)

    d1, _ = samples.shape

    if diffpriv:
        Z = samples[np.random.randint(0, d1)]
        delta = -alpha * (g + (1 / batch_size) * Z)
    else:
        delta = -alpha * g

    # Weird way to get NON top k values
    if theta < 1:
        param_filter = np.argpartition(
            abs(delta), -threshold)[:d - threshold]
        delta[param_filter] = 0

    w_new = ww + delta
    f_new, g_new = funObj(w_new, X[idx, :], y[idx], batch_size)
    iteration = iteration + 1

    return delta
