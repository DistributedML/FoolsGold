import numpy as np 
import pandas as pd
import arff
import pdb

def slice_iid():

    data = np.load('amazon.npy')

    for numclassesper in np.arange(5, 50, 5):

        print("Slicing up for " + str(numclassesper))

        # Shuffle the data
        sfflidx = np.random.permutation(data.shape[0])
        data = data[sfflidx]

        testidx = int(data.shape[0] * 0.7)

        testdata = data[testidx:, ]
        traindata = data[0:testidx, ]

        # standardize each column
        traindata[:, 0:10000], _, _ = standardize_cols(traindata[:, 0:10000])
        testdata[:, 0:10000], _, _ = standardize_cols(testdata[:, 0:10000])

        for k in range(int(np.max(traindata[:, 10000]) + 1)):

            filesuf = ""
            idx_bool = np.full(traindata.shape[0], False)
            
            for i in range(numclassesper):
                idx_bool += traindata[:, 10000] == ((k + i) % 50)
                filesuf += "_" + str((k + i) % 50)
            
            idx = np.where(idx_bool)[0]
            
            print("Label " + filesuf + " has " + str(len(idx)))
            labeldata = traindata[idx]

            np.save("amazon" + filesuf, labeldata)


def main():

    data = np.load('amazon.npy')

    # Shuffle the data
    sfflidx = np.random.permutation(data.shape[0])
    data = data[sfflidx]

    testidx = int(data.shape[0] * 0.7)

    testdata = data[testidx:, ]
    traindata = data[0:testidx, ]

    # standardize each column
    traindata[:, 0:10000], _, _ = standardize_cols(traindata[:, 0:10000])
    testdata[:, 0:10000], _, _ = standardize_cols(testdata[:, 0:10000])

    for i in range(int(np.max(traindata[:, 10000]) + 1)):

        idx = np.where(traindata[:, 10000] == i)[0]
        print("Label " + str(i) + " has " + str(len(idx)))
        
        labeldata = traindata[idx]

        np.save("amazon_" + str(i), labeldata)

    np.save("amazon_train", traindata)
    np.save("amazon_test", testdata)


def load_raw():

    datadump = arff.load(open('amazon.arff', 'rb'))
    data = np.array(datadump['data'])

    # Convert labels to categorical
    data[:, -1] = np.argmax(pd.get_dummies(data[:, -1]).values, axis=1)
    data = data.astype(float)

    pdb.set_trace()

    np.save("amazon", data)


def standardize_cols(X, mu=None, sigma=None):
    # Standardize each column with mean 0 and variance 1
    n_rows, n_cols = X.shape

    if mu is None:
        mu = np.mean(X, axis=0)

    if sigma is None:
        sigma = np.std(X, axis=0)
        sigma[sigma < 1e-8] = 1.

    return (X - mu) / sigma, mu, sigma


if __name__ == "__main__":

    slice_iid()
