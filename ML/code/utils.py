from __future__ import division
import pickle
import os
import sys
import numpy as np
from numpy.linalg import norm
import pandas as pd
import pdb

def sliceup(numSplits, dataset):

    data = load_dataset(dataset)

    X, y = data['X'], data['y']
    #Xvalid, yvalid = data['Xvalid'], data['yvalid']

    randseed = np.random.permutation(X.shape[0])
    X = X[randseed, :]
    y = y[randseed]

    numRows = int(X.shape[0] / numSplits)
    np.savetxt("../data/" + dataset + "train.csv", 
        np.hstack((X, y.reshape((X.shape[0], 1)))), 
        delimiter=",")

    for i in range(numSplits):
        dataslice = np.hstack((X[(i * numRows):((i + 1) * numRows), :],
                        y[(i * numRows):((i + 1) * numRows)].reshape((numRows, 1))))
        np.savetxt("../data/" + dataset + str(i + 1) + ".csv", dataslice, delimiter=",")

    #numTestRows = Xvalid.shape[0]
    #datatest = np.hstack((Xvalid, yvalid.reshape((numTestRows, 1))))
    #np.savetxt("../data/" + dataset + "test.csv", datatest, delimiter=",")

def bootstrap(numSets, dataset):

    data = load_dataset(dataset)
    X, y = data['X'], data['y']
    Xvalid, yvalid = data['Xvalid'], data['yvalid']

    nn = X.shape[0]

    # Sample n points with replacement from n examples
    for i in range(numSets):
        bootsample = np.random.choice(nn, nn)
        Xboot = X[bootsample, :]
        yboot = y[bootsample]

        dataslice = np.hstack((Xboot, yboot.reshape((nn, 1))))
        np.savetxt("../bootstraps/" + dataset + "_boot_" + 
            str(i + 1) + "_g.csv", dataslice, delimiter=",")

        dataslice = np.hstack((Xboot, (yboot * -1).reshape((nn, 1))))
        np.savetxt("../bootstraps/" + dataset + "_boot_" + 
            str(i + 1) + "_b.csv", dataslice, delimiter=",")


def load_dataset(dataset_name, npy=False):

    # Load and standardize the data and add the bias term
    if dataset_name == "logisticData":
        # data = load_pkl(os.path.join('..', "data", 'logisticData.pkl'))
        # Made change here
        data = load_pkl(os.path.join('../ML', "data", 'logisticData.pkl'))

        X, y = data['X'], data['y']
        Xvalid, yvalid = data['Xvalidate'], data['yvalidate']

        n, _ = X.shape

        randseed = np.random.permutation(n)
        X = X[randseed, :]
        y = y[randseed]

        X, mu, sigma = standardize_cols(X)
        Xvalid, _, _ = standardize_cols(Xvalid, mu, sigma)

        X = np.hstack([np.ones((X.shape[0], 1)), X])
        Xvalid = np.hstack([np.ones((Xvalid.shape[0], 1)), Xvalid])

        X = normalize_rows(X)
        Xvalid = normalize_rows(Xvalid)

        return {"X": X, "y": y,
                "Xvalid": Xvalid,
                "yvalid": yvalid}

    elif dataset_name == "credit":

        df = pd.read_csv(os.path.join('../ML', "data", 'creditcard.csv'))
        nn, dd = df.shape

        # Need to remove the first column and first row
        credit = df.ix[1:nn, 1:dd].as_matrix()
        nn, dd = credit.shape

        datay = credit[:, dd - 1].astype(int)
        datay[np.where(datay == 0)] = -1

        idx = np.arange(nn)
        data = credit[idx, :].astype(float)

        split = int(nn * 0.70)
        X = data[0:split, :]
        y = datay[0:split]

        Xvalid = data[split:nn - 1, :]
        yvalid = datay[split:nn - 1]

        X, mu, sigma = standardize_cols(X)
        Xvalid, _, _ = standardize_cols(Xvalid, mu, sigma)

        X = np.hstack([np.ones((X.shape[0], 1)), X])
        Xvalid = np.hstack([np.ones((Xvalid.shape[0], 1)), Xvalid])

        #X = normalize_rows(X)
        #Xvalid = normalize_rows(Xvalid)

        return {"X": X, "y": y,
                "Xvalid": Xvalid,
                "yvalid": yvalid}

    elif dataset_name == "slices":

        slices = pd.read_csv(os.path.join(
            '../', "data", 'slice_localization_data.csv'))
        n, d = slices.shape

        npslices = slices.ix[np.random.permutation(n), :].as_matrix()
        split = int(n * 0.70)

        X = npslices[0:split, 0:d - 1]
        y = npslices[0:split, -1]

        Xvalid = npslices[(split + 1):n, 0:d - 1]
        yvalid = npslices[(split + 1):n, -1]

        X, mu, sigma = standardize_cols(X)
        Xvalid, _, _ = standardize_cols(Xvalid, mu, sigma)

        X = np.hstack([np.ones((X.shape[0], 1)), X])
        Xvalid = np.hstack([np.ones((Xvalid.shape[0], 1)), Xvalid])

        return {"X": X, "y": y,
                "Xvalid": Xvalid,
                "yvalid": yvalid}

    elif dataset_name == "magic":

        magic = pd.read_csv(os.path.join('../ML', "data", 'magic04.data'))
        nn, dd = magic.shape

        y = magic.ix[:, dd - 1].as_matrix()
        y[np.where(y == 'g')] = 1
        y[np.where(y == 'h')] = -1

        npmagic = magic.ix[np.random.permutation(
            nn), :].as_matrix().astype(float)
        split = int(nn * 0.70)

        # Add another column for the square of each value
        data = npmagic[:, 0:dd-1]
        data = np.hstack([data, (data * data)])

        X = data[0:split, :]
        y = npmagic[0:split, dd - 1]

        Xvalid = data[split:nn - 1]
        yvalid = npmagic[split:nn - 1, dd - 1]

        X, mu, sigma = standardize_cols(X)
        Xvalid, _, _ = standardize_cols(Xvalid, mu, sigma)

        X = np.hstack([np.ones((X.shape[0], 1)), X])
        Xvalid = np.hstack([np.ones((Xvalid.shape[0], 1)), Xvalid])

        X = normalize_rows(X)
        Xvalid = normalize_rows(Xvalid)

        return {"X": X, "y": y,
                "Xvalid": Xvalid,
                "yvalid": yvalid}

    elif dataset_name == "wdbc":

        wdbc = pd.read_csv(os.path.join('../', "data", 'wdbc.data'))
        nn, dd = wdbc.shape

        # First column is ID. Second one is diagnosis 'M' or 'B'
        y = wdbc.ix[:, 1].as_matrix()
        y[np.where(y == 'M')] = 1
        y[np.where(y == 'B')] = -1

        npbc = wdbc.ix[np.random.permutation(
            nn), :].as_matrix().astype(float)
        split = int(nn * 0.70)

        data = npbc[:, 2:dd]

        X = data[0:split, :]
        y = npbc[0:split, 1]

        Xvalid = data[split:nn]
        yvalid = npbc[split:nn, 1]

        X, mu, sigma = standardize_cols(X)
        Xvalid, _, _ = standardize_cols(Xvalid, mu, sigma)

        X = np.hstack([np.ones((X.shape[0], 1)), X])
        Xvalid = np.hstack([np.ones((Xvalid.shape[0], 1)), Xvalid])

        return {"X": X, "y": y,
                "Xvalid": Xvalid,
                "yvalid": yvalid}

    else:

        # This is the main section typically called by the tor client.
        # Thus we have hardcoded an absolute path from that location

        if npy:

            data = np.load(os.path.join('../ML', "data", dataset_name + '.npy'))
            n, d = data.shape

        else:

            data = pd.read_csv(os.path.join('../ML', "data", dataset_name + '.csv'))
            d = data.shape[1]

            data = data.as_matrix()

        X = data[:, 0:d - 1]
        y = data[:, -1]

        return {"X": X, "y": y}


def normalize_rows(X):

    # Sets the max row to have L2 norm of 1. Needed for diff priv
    nn, dd = X.shape
    max_norm = 0

    for i in xrange(nn):
        new_norm = norm(X[i, ], 2)
        if new_norm > max_norm:
            max_norm = new_norm

    X = X / max_norm

    return X


def standardize_cols(X, mu=None, sigma=None):
    # Standardize each column with mean 0 and variance 1
    n_rows, n_cols = X.shape

    if mu is None:
        mu = np.mean(X, axis=0)

    if sigma is None:
        sigma = np.std(X, axis=0)
        sigma[sigma < 1e-8] = 1.

    return (X - mu) / sigma, mu, sigma


def standardize_outputs(y, mu=None, sigma=None):

    if mu is None:
        mu = np.mean(y)

    if sigma is None:
        sigma = np.std(y)
        if sigma < 1e-8:
            sigma = 1.

    return (y - mu) / sigma, mu, sigma


def check_gradient(model, X, y):
    # This checks that the gradient implementation is correct
    w = np.random.rand(model.w.size)
    f, g = model.funObj(w, X, y)

    # Check the gradient
    estimated_gradient = approx_fprime(w,
                                       lambda w: model.funObj(w, X, y)[0],
                                       epsilon=1e-6)

    implemented_gradient = model.funObj(w, X, y)[1]

    if np.max(np.abs(estimated_gradient - implemented_gradient) > 1e-4):
        raise Exception('User and numerical derivatives differ:\n%s\n%s' %
                        (estimated_gradient[:5], implemented_gradient[:5]))
    else:
        print('User and numerical derivatives agree.')


def lap_noise(loc=0, scale=1, size=1):
    return np.random.laplace(loc=loc, scale=scale, size=size)


def exp_noise(scale=1, size=1):
    return np.random.exponential(scale=scale, size=size)


def approx_fprime(x, f_func, epsilon=1e-7):
    # Approximate the gradient using the complex step method
    n_params = x.size
    e = np.zeros(n_params)
    gA = np.zeros(n_params)
    for n in range(n_params):
        e[n] = 1.
        val = f_func(x + e * np.complex(0, epsilon))
        gA[n] = np.imag(val) / epsilon
        e[n] = 0

    return gA


def regression_error(y, yhat):
    return 0.5 * np.sum(np.square((y - yhat)) / float(yhat.size))


def classification_error(y, yhat):
    return np.sum(y != yhat) / float(yhat.size)


def load_pkl(fname):
    """Reads a pkl file.

    Parameters
    ----------
    fname : the name of the .pkl file

    Returns
    -------
    data :
        Returns the .pkl file as a 'dict'
    """
    if not os.path.isfile(fname):
        raise ValueError('File {} does not exist.'.format(fname))

    if sys.version_info[0] < 3:
        # Python 2
        with open(fname, 'rb') as f:
            data = pickle.load(f)
    else:
        # Python 3
        with open(fname, 'rb') as f:
            data = pickle.load(f, encoding='latin1')

    return data
