import numpy as np
import pandas as pd
import pdb


def eval(Xtest, ytest, weights, correctLabel, targetLabel, 
    numClasses, numFeatures, verbose=True):

    # hardcoded for MNIST
    W = np.reshape(weights, (numClasses, numFeatures))
    yhat = np.argmax(np.dot(Xtest, W.T), axis=1)

    targetIdx = np.where(ytest == correctLabel)
    otherIdx = np.where(ytest != correctLabel)
    overall = np.mean(yhat == ytest)
    others = np.mean(yhat[otherIdx] == ytest[otherIdx])
    correct1 = np.mean(yhat[targetIdx] == correctLabel)
    attacked1 = np.mean(yhat[targetIdx] == targetLabel)

    targetlabel_idx = np.where(ytest == targetLabel)
    targetlabel_correct = np.mean(yhat[targetlabel_idx] == targetLabel)
    
    if verbose:
        print("Accuracy overall: " + str(overall))
        print("Accuracy on other digits: " + str(others))
        print("Target Accuracy on source label " + str(correctLabel) + "s: " + str(correct1))
        print("Target Accuracy on target label " + str(targetLabel) + "s: " + str(targetlabel_correct))
        print("Target Attack Rate (" + str(correctLabel) + " to " + str(targetLabel) + "): " + str(attacked1)  + "\n")
    else:
        print("Accuracy overall: " + str(overall))
        print("Target Attack Rate (" + str(correctLabel) + " to " + str(targetLabel) + "): " + str(attacked1)  + "\n")

    return overall, others, correct1, targetlabel_correct, attacked1
    
def backdoor_eval(Xtest, ytest, weights, targetLabel, 
    numClasses, numFeatures, verbose=True):

     # hardcoded for MNIST
    W = np.reshape(weights, (numClasses, numFeatures))
    yhat = np.argmax(np.dot(Xtest, W.T), axis=1)

    # All digits where the bottom right has value > 1
    targetIdx = np.where(Xtest[:,783] > 1)
    
    overall = np.mean(yhat == ytest)
    correct1 = np.mean(yhat[targetIdx] == ytest[targetIdx])
    attacked1 = np.mean(yhat[targetIdx] == targetLabel)

    print("Accuracy overall: " + str(overall))
    print("Target Accuracy backdoored examples: " + str(correct1))
    print("Target Attack Rate (Backdoored to " + str(targetLabel) + "): " + str(attacked1)  + "\n")
    
    return overall, correct1, attacked1


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
