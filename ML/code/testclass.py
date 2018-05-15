from __future__ import division
import numpy as np
import utils
import pdb

verbose = 0
maxEvals = 100
X = utils.load_dataset("slices")['X']
y = utils.load_dataset("slices")['y']
alpha = 1
d = X.shape[1]
w = np.zeros(d)
lammy = 0.1

class linReg:

    def __init__(self, verbose, maxEvals):
        self.verbose = verbose
        self.maxEvals = maxEvals

    def changeVerbose(self, verbose=0):
        self.verbose = verbose