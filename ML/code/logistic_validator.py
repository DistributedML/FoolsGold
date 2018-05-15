from __future__ import division
import numpy as np
import utils
import pdb

data = utils.load_dataset("credittest")
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
    
    ww = np.array(ww)
    yhat = np.sign(np.dot(Xvalid, ww))
    
    ww2 = np.array(ww + delta)
    yhat2 = np.sign(np.dot(Xvalid, ww2))

    g_err = np.sum(yhat != yvalid) / float(yvalid.size)
    new_err = np.sum(yhat2 != yvalid) / float(yvalid.size)

    return new_err - g_err

if __name__ == "__main__":
	pdb.set_trace()