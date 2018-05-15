from __future__ import division
from numpy.linalg import norm
from scipy.misc import logsumexp
import numpy as np
import utils
import pdb
import emcee

iteration = 1
alpha = 1e-2

hist_grad = 0
epsilon = 0

scale = False
diffpriv = False


class SoftMaxModel:

    def __init__(self, dataset, epsilon, numClasses):

        passedEpsilon = epsilon
        data = utils.load_dataset(dataset, npy=True)

        self.X = data['X']

        self.y = data['y']
        self.n_classes = numClasses
        #self.n_classes = 23
        # Different for softmax
        self.d = self.X.shape[1] * self.n_classes
        self.samples = []
        self.lammy = 0.01

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


    def get_data(self):
        return self.X, self.y

    def funObj(self, ww, Xbatch, ybatch, batch_size):
        n, d = Xbatch.shape

        W = np.reshape(ww, (self.n_classes, d))

        y_binary = np.zeros((n, self.n_classes)).astype(bool)
        y_binary[np.arange(n), ybatch.astype(int)] = 1

        XW = np.dot(Xbatch, W.T)
        #XW = XW - np.max(XW)
        #Z = np.sum(np.exp(XW), axis=1)
        # Calculate the function value
        f = - np.sum(XW[y_binary] - logsumexp(XW))
        #
        # Calculate the gradient value

        mval = np.max(XW)
        XW = XW - mval
        Z = np.sum(np.exp(XW), axis=1)
        v = np.exp(XW) / Z[:, None]
        v[np.isnan(v)] = 0
        res = np.dot((v - y_binary).T, Xbatch)

        # Some DP methods only work if the gradient is scaled down to have a max norm of 1
        if scale:
            g = (1 / batch_size) * res / max(1, np.linalg.norm(res)) + self.lammy * W
        else:
            g = (1 / batch_size) * res + self.lammy * W
        if True in np.isnan(g):
            pdb.set_trace()
        return f, g.flatten()


    # def funObj(self, ww, Xbatch, ybatch, batch_size):
    #
    #     n, d = Xbatch.shape
    #
    #     W = np.reshape(ww, (self.n_classes, d))
    #
    #     y_binary = np.zeros((n, self.n_classes)).astype(bool)
    #     y_binary[np.arange(n), ybatch.astype(int)] = 1
    #
    #     XW = np.dot(Xbatch, W.T)
    #     Z = np.sum(np.exp(XW), axis=1)
    #
    #     # Calculate the function value
    #     f = - np.sum(XW[y_binary] - np.log(Z))
    #
    #     # Calculate the gradient value
    #     res = np.dot((np.exp(XW) / Z[:, None] - y_binary).T, Xbatch)
    #
    #     # Some DP methods only work if the gradient is scaled down to have a max norm of 1
    #     if scale:
    #         g = (1 / batch_size) * res / max(1, np.linalg.norm(res)) + self.lammy * W
    #     else:
    #         g = (1 / batch_size) * res + self.lammy * W
    #
    #     return f, g.flatten()



    # Reports the direct change to w, based on the given one.
    # Batch size could be 1 for SGD, or 0 for full gradient.
    def privateFun(self, theta, ww, batch_size=0):

        ww = np.array(ww)

        # Define constants and params
        nn, dd = self.X.shape

        if batch_size > 0 and batch_size < nn:
            idx = np.random.choice(nn, batch_size, replace=False)
        else:
            # Just take the full range
            idx = range(nn)

        f, g = self.funObj(ww, self.X[idx, :], self.y[idx], batch_size)

        if diffpriv:
            d1, _ = self.samples.shape
            Z = self.samples[np.random.randint(0, d1)]
            delta = -alpha * (g + (1 / batch_size) * Z)
        else:
            delta = -alpha * g

        # w_new = ww + delta
        # f_new, g_new = funObj(w_new, X[idx, :], y[idx], batch_size)

        return delta
