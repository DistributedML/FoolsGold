import numpy as np 
import pandas as pd
import arff
import pdb


def main():

    data = np.load('amazon.npy')

    # Shuffle the data
    sfflidx = np.random.permutation(data.shape[0])
    data = data[sfflidx]

    # testidx = int(data.shape[0] * 0.7)

    # testdata = data[testidx:, ]
    # traindata = data[0:testidx, ]

    # standardize each column
    data[:, 0:10000], _, _ = standardize_cols(data[:, 0:10000])
    # testdata[:, 0:10000], _, _ = standardize_cols(testdata[:, 0:10000])

    for i in range(int(np.max(data[:, 10000]) + 1)):

        idx = np.where(data[:, 10000] == i)[0]
        print("Label " + str(i) + " has " + str(len(idx)))

        labeldata1 = data[idx[0:15]]
        labeldata2 = data[idx[15:30]]

        np.save("amazon" + str(i) + "a", labeldata1)
        np.save("amazon" + str(i) + "b", labeldata2)

    np.save("amazon_all", data)
    # np.save("amazon_test", testdata)

    # Make a bad dataset, push class 0 to 11 (normal)
    # baddata = traindata[np.where(traindata[:, 41] == 0)[0]]
    # baddata[:, -1] = 11

    # np.save("amazon_bad", baddata)


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

    main()
