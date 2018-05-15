import numpy as np
import pandas as pd
import pdb


def eval(Xtest, ytest, weights, correctLabel, missLabel, numClasses, numFeatures):

    # hardcoded for MNIST
    W = np.reshape(weights, (numClasses, numFeatures))
    yhat = np.argmax(np.dot(Xtest, W.T), axis=1)

    targetIdx = np.where(ytest == correctLabel)
    otherIdx = np.where(ytest != correctLabel)
    overall = np.mean(yhat[otherIdx] == ytest[otherIdx])
    correct1 = np.mean(yhat[targetIdx] == correctLabel)
    attacked1 = np.mean(yhat[targetIdx] == missLabel)

    misslabel_idx = np.where(ytest == missLabel)
    misslabel_correct = np.mean(yhat[misslabel_idx] == missLabel)

    print("Overall Error: " + str(overall))
    print("Target Training Accuracy on " + str(correctLabel) + "s: " + str(correct1))
    print("Target Training Accuracy on misslabel " + str(missLabel) + "s: " + str(misslabel_correct))
    print("Target Attack Rate (" + str(correctLabel) + " to " + str(missLabel) + "): " + str(attacked1)  + "\n")
    return overall, correct1, misslabel_correct, attacked1



def main():

    dataTrain = np.load("../data/mnist_train.npy")
    dataTest = np.load("../data/mnist_train.npy")

    df = pd.read_csv("../../DistSys/modelflush_pure.csv", header=None)
    pure_model = df.ix[0, :7839].as_matrix().astype(float)

    Xtrain = dataTest[:, :784]
    ytrain = dataTest[:, 784]

    eval(Xtrain, ytrain, pure_model)

    df = pd.read_csv("../../DistSys/modelflush_1p.csv", header=None)
    poison_model = df.ix[0, :7839].as_matrix().astype(float)

    eval(Xtrain, ytrain, poison_model)

    df = pd.read_csv("../../DistSys/modelflush_2p.csv", header=None)
    poison_model = df.ix[0, :7839].as_matrix().astype(float)

    eval(Xtrain, ytrain, poison_model)

    pdb.set_trace()


if __name__ == "__main__":

    main()
