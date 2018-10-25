from __future__ import division
from numpy.linalg import norm
from scipy.misc import logsumexp
import numpy as np
import utils
import pdb

iteration = 1

# learning rate
alpha = 1e-2

class SoftMaxModel:

    def __init__(self, dataset, numClasses):

        data = utils.load_dataset(dataset, npy=True)

        self.X = data['X']

        self.y = data['y']
        self.n_classes = numClasses
        self.d = self.X.shape[1] * self.n_classes
        self.samples = []
        self.lammy = 0.01

        def lnprob(x, alpha):
            return -(alpha / 2) * np.linalg.norm(x)

    def get_data(self):
        return self.X, self.y

    def funObj(self, ww, Xbatch, ybatch, batch_size):
        n, d = Xbatch.shape

        W = np.reshape(ww, (self.n_classes, d))

        y_binary = np.zeros((n, self.n_classes)).astype(bool)
        y_binary[np.arange(n), ybatch.astype(int)] = 1

        XW = np.dot(Xbatch, W.T)
        
        # Calculate the function value
        f = - np.sum(XW[y_binary] - logsumexp(XW))
        
        # Calculate the gradient value
        mval = np.max(XW)
        XW = XW - mval
        Z = np.sum(np.exp(XW), axis=1)
        v = np.exp(XW) / Z[:, None]
        v[np.isnan(v)] = 0
        res = np.dot((v - y_binary).T, Xbatch)

        g = (1 / batch_size) * res + self.lammy * W
        
        if True in np.isnan(g):
            pdb.set_trace()

        return f, g.flatten()


    def privateFun(self, ww, batch_size=0, num_iterations=1):
        
        '''
        Reports the direct change to be applied, based on the given model.
        Batch size could be 1 for SGD, or 0 for full gradient.
        Specify the number of iterations for batching multiple iterations as in FEDAVG
        '''

        # Take the total gradient locally over multiple iterations
        ww = np.array(ww)
        total_delta = np.zeros(ww.shape[0])

        for it in range(num_iterations):

            # Define constants and params
            nn, dd = self.X.shape

            if batch_size > 0 and batch_size < nn:
                idx = np.random.choice(nn, batch_size, replace=False)
            else:
                # Just take the full range
                idx = range(nn)

            f, g = self.funObj(ww, self.X[idx, :], self.y[idx], batch_size)
            delta = -alpha * g

        ww = ww + delta
        total_delta = total_delta + delta

        return total_delta



class SoftMaxModelEvil(SoftMaxModel):

    '''
    A softmax model that conforms to the API, 
    but actually just sends the vector to the optimal poisoning model
    '''

    def __init__(self, dataset, numClasses):

        SoftMaxModel.__init__(self, dataset, numClasses)

        batch_size = 5
        iterations = 3000

        # Just train the goal poisoned model locally
        self.weights = np.random.rand(self.d) / 100.0

        for i in xrange(iterations):

            if batch_size > 0 and batch_size < self.X.shape[0]:
                idx = np.random.choice(self.X.shape[0], batch_size, replace=False)
            else:
                # Just take the full range
                idx = range(self.X.shape[0])

            f, g = self.funObj(self.weights, self.X[idx, :], self.y[idx],
               batch_size)
            delta = -alpha * g
            self.weights = self.weights + delta

    def privateFun(self, theta, ww, batch_size=0):

        ww = np.array(ww)

        # Send the vector that just moves the model to the poisoning goal
        delta = self.weights - ww

        return delta
    
