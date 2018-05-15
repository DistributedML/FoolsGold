import pandas as pd
import pdb
import numpy as np

def main():
    df = pd.read_csv("kddcup.csv", header=None)

    for i in [1, 2, 3, 41]:
        df[i] = df[i].astype("category").cat.codes

    data = df.as_matrix()
    sfflidx = np.random.permutation(data.shape[0])

    # Shuffle the data
    data = data[sfflidx]

    print(data.shape)

    testidx = int(data.shape[0] * 0.7)

    testdata = data[testidx:, ]
    traindata = data[0:testidx, ]

    # standardize each column
    traindata[:, 0:41], _, _ = standardize_cols(traindata[:, 0:41])
    testdata[:, 0:41], _, _ = standardize_cols(testdata[:, 0:41])

    for i in range(int(np.max(data[:, 41]) + 1)):

        idx = np.where(traindata[:, 41] == i)[0]
        print("Label " + str(i) + " has " + str(len(idx)))
        labeldata = traindata[idx]
        np.save("kddcup" + str(i), labeldata)

        ovridx = np.where(data[:, 41] == i)[0]
        print("Overall, label " + str(i) + " has " + str(len(ovridx)))

    np.save("kddcup_train", traindata)
    np.save("kddcup_test", testdata)

    pdb.set_trace()

    # Make a bad dataset, push class 0 to 11 (normal)
    baddata = traindata[np.where(traindata[:, 41] == 0)[0]]
    baddata[:, -1] = 11

    pdb.set_trace()

    np.save("kddcup_bad", baddata)

# pd.get_dummies(df, columns=[1, 2, 3])


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
