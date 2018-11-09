from mnist import MNIST
from sklearn import svm, linear_model, neural_network
import pdb
import numpy as np
import matplotlib.pyplot as plt

#################################################################
#### IMPORTANT!!!
#### Download and gunzip files from http://yann.lecun.com/exdb/mnist/
#################################################################

def main():

    mndata = MNIST('.')

    images, labels = mndata.load_training()
    images_test, labels_test = mndata.load_testing()

    n = len(images)
    d = len(images[0])
    t = len(images_test)

    Xtrain = np.zeros((n, d))
    Xtest = np.zeros((t, d))

    ytrain = np.asarray(labels)
    ytest = np.asarray(labels_test)

    for i in range(n):
        Xtrain[i, :] = np.asarray(images[i])

    for q in range(t):
        Xtest[q, :] = np.asarray(images_test[q])

    print("Training classifier.")

    clf = linear_model.SGDClassifier(loss='log', max_iter=1000, tol=0.01)
    clf.fit(Xtrain, ytrain)

    # Training error
    y_hat = clf.predict(Xtrain)
    train_error = np.mean(y_hat != ytrain)
    print("Training Err: " + str(train_error))

    y_hat_test = clf.predict(Xtest)
    test_error = np.mean(y_hat_test != ytest)
    print("Test Err: " + str(test_error))

    nn = neural_network.MLPClassifier()
    nn.fit(Xtrain, ytrain)

    # Training error
    y_hat = nn.predict(Xtrain)
    train_error = np.mean(y_hat != ytrain)
    print("Training Err: " + str(train_error))

    y_hat_test = nn.predict(Xtest)
    test_error = np.mean(y_hat_test != ytest)
    print("Test Err: " + str(test_error))


def slice_federated():
    '''
    Each file contains one of the 10 classes
    '''
    mndata = MNIST('.')

    images, labels = mndata.load_training()
    images_test, labels_test = mndata.load_testing()

    n = len(images)
    d = len(images[0])
    t = len(images_test)

    Xtrain = np.zeros((n, d))
    Xtest = np.zeros((t, d))

    ytrain = np.asarray(labels)
    ytest = np.asarray(labels_test)

    for i in range(n):
        Xtrain[i, :] = np.asarray(images[i])

    for q in range(t):
        Xtest[q, :] = np.asarray(images_test[q])

    # standardize each column
    print("Standardize columns")
    Xtrain = Xtrain / 100.0
    # Xtrain, _, _ = standardize_cols(Xtrain)
    # Xtest, _, _ = standardize_cols(Xtest)

    for k in range(10):

        idx = np.where((ytrain == k))[0]

        class_slice = Xtrain[idx]
        data_slice = np.hstack((class_slice, ytrain[idx][:, None]))

        print("slice " + str(k) + " is shape " + str(data_slice.shape))

        np.save("mnist" + str(k), data_slice)

    train_slice = np.hstack((Xtrain, np.reshape(ytrain, (len(ytrain), 1))))
    np.save("mnist_train", train_slice)

    test_slice = np.hstack((Xtest, np.reshape(ytest, (len(ytest), 1))))
    np.save("mnist_test", test_slice)


def slice_uniform(numSplits):
    '''
    Each file contains a horizontal partition of the dataset
    '''
    mndata = MNIST('.')

    images, labels = mndata.load_training()
    images_test, labels_test = mndata.load_testing()

    n = len(images)
    d = len(images[0])
    t = len(images_test)

    Xtrain = np.zeros((n, d))
    Xtest = np.zeros((t, d))

    ytrain = np.asarray(labels)
    ytest = np.asarray(labels_test)

    for i in range(n):
        Xtrain[i, :] = np.asarray(images[i])

    for q in range(t):
        Xtest[q, :] = np.asarray(images_test[q])

    # standardize each column
    print("Standardize columns")
    Xtrain = Xtrain / 100.0

    randseed = np.random.permutation(n)
    Xtrain = Xtrain[randseed, :]
    ytrain = ytrain[randseed]

    numRows = int(n / numSplits)

    for i in range(numSplits):
        
        dataslice = np.hstack((Xtrain[(i * numRows):((i + 1) * numRows), :],
                        ytrain[(i * numRows):((i + 1) * numRows)].reshape((numRows, 1))))
        
        print("slice " + str(i) + " is shape " + str(dataslice.shape))

        np.save("mnist_uniform_" + str(i), dataslice)

    train_slice = np.hstack((Xtrain, np.reshape(ytrain, (len(ytrain), 1))))
    np.save("mnist_train", train_slice)

    test_slice = np.hstack((Xtest, np.reshape(ytest, (len(ytest), 1))))
    np.save("mnist_test", test_slice)


def slice_classmix(nclassesper):
    '''
    Each file contains multiple classes
    nclassesper - number of classes to include per file
    '''

    mndata = MNIST('.')

    images, labels = mndata.load_training()
    images_test, labels_test = mndata.load_testing()

    n = len(images)
    d = len(images[0])
    t = len(images_test)

    Xtrain = np.zeros((n, d))
    Xtest = np.zeros((t, d))

    ytrain = np.asarray(labels)
    ytest = np.asarray(labels_test)

    for i in range(n):
        Xtrain[i, :] = np.asarray(images[i])

    for q in range(t):
        Xtest[q, :] = np.asarray(images_test[q])

    # standardize each column
    print("Standardize columns")
    Xtrain = Xtrain / 100.0
    # Xtrain, _, _ = standardize_cols(Xtrain)
    # Xtest, _, _ = standardize_cols(Xtest)

    for k in range(10):

        filesuf = ""
        idx_bool = np.full(len(ytrain), False)
        for i in range(nclassesper):
            idx_bool += (ytrain == (k + i) % 10)
            filesuf += str((k + i) % 10)

        idx = np.where(idx_bool)[0]

        class_slice = Xtrain[idx]
        data_slice = np.hstack((class_slice, ytrain[idx][:, None]))

        print("slice " + filesuf + " is shape " + str
            (data_slice.shape))

        np.save("mnist" + filesuf, data_slice)

    train_slice = np.hstack((Xtrain, np.reshape(ytrain, (len(ytrain), 1))))
    np.save("mnist_train", train_slice)

    test_slice = np.hstack((Xtest, np.reshape(ytest, (len(ytest), 1))))
    np.save("mnist_test", test_slice)


def show_digit(image):

    plt.imshow(image, cmap='gray')
    plt.show()


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

    slice_federated()
    slice_uniform(10)
