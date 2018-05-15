from __future__ import division
import numpy as np
import utils
import pdb

Xtest = utils.load_dataset('linTest')['X']
ytest = utils.load_dataset('linTest')['y']


def test(ww):
    ww = np.array(ww)
    yhat = np.dot(Xtest, ww)
    error = 0.5 * np.sum(np.square((ytest - yhat)) / float(yhat.size))
    return error
