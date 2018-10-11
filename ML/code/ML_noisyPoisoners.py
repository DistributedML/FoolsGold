from __future__ import division
from numpy.linalg import norm
import matplotlib.pyplot as plt

import model_aggregator
import softmax_model_test
import softmax_model_obj
import poisoning_compare

import numpy as np
import utils

import pdb
import sys


np.set_printoptions(suppress=True)

# Just a simple sandbox for testing out python code, without using Go.
def debug_signal_handler(signal, frame):
    import pdb
    pdb.set_trace()


import signal
signal.signal(signal.SIGINT, debug_signal_handler)


def basic_conv(dataset, num_params, softmax_test, iterations=3000):

    batch_size = 5

    # Global
    # numFeatures = softmax_model.init(dataset, epsilon=epsilon)
    softmax_model = softmax_model_obj.SoftMaxModel(dataset, numClasses)

    print("Start training")

    weights = np.random.rand(num_params) / 100.0

    train_progress = np.zeros(iterations)
    test_progress = np.zeros(iterations)

    for i in xrange(iterations):
        deltas = softmax_model.privateFun(1, weights, batch_size)
        weights = weights + deltas

        if i % 100 == 0:
            print("Train error: %.10f" % softmax_test.train_error(weights))

    print("Done iterations!")
    print("Train error: %d", softmax_test.train_error(weights))
    print("Test error: %d", softmax_test.test_error(weights))
    return weights

def rescale(x, a, b):
    minNum = np.min(x)
    maxNum = np.max(x)
    return (b - a)*(x - minNum) / (maxNum - minNum) + a 

def cos(vecA, vecB):
    return np.dot(vecA, vecB)/(np.linalg.norm(vecA) * np.linalg.norm(vecB))

# Variant of non_iid, where noise is added to poisoner_indices
def non_iid(model_names, numClasses, numParams, softmax_test, topk_prop, iterations=3000, numSybils=2,
    ideal_attack=False, poisoner_indices = []):

    batch_size = 50
    topk = int(numParams / 10)

    list_of_models = []

    for dataset in model_names:
        list_of_models.append(softmax_model_obj.SoftMaxModel(dataset, numClasses))

    # Include the model that sends the ideal vector on each iteration
    if ideal_attack:
        list_of_models.append(softmax_model_obj.SoftMaxModelEvil(dataPath +
           "_bad_ideal_4_9", numClasses))

    numClients = len(list_of_models)
    model_aggregator.init(numClients, numParams, numClasses)

    print("Start training across " + str(numClients) + " clients.")

    weights = np.random.rand(numParams) / 100.0
    train_progress = []

    summed_deltas = np.zeros((numClients, numParams))

    for i in xrange(iterations):

        delta = np.zeros((numClients, numParams))
        
        # Significant features filter
        # sig_features_idx = np.argpartition(weights, -topk)[-topk:]
        sig_features_idx = np.arange(numParams)

        for k in range(len(list_of_models)):
            delta[k, :] = list_of_models[k].privateFun(1, weights, batch_size)

            # normalize delta
            if np.linalg.norm(delta[k, :]) > 1:
                delta[k, :] = delta[k, :] / np.linalg.norm(delta[k, :])

        # Add adversarial noise
        noisevec = rescale(np.random.rand(numParams), np.min(delta), np.max(delta))
        delta[poisoner_indices[0], :] = delta[poisoner_indices[0], :] + noisevec
        delta[poisoner_indices[1], :] = delta[poisoner_indices[1], :] - noisevec
        
        # Track the total vector from each individual client
        summed_deltas = summed_deltas + delta
        
        # Use Foolsgold
        this_delta = model_aggregator.foolsgold(delta, summed_deltas, sig_features_idx, i, weights, topk_prop, importance=False, importanceHard=True)
        # this_delta = model_aggregator.average(delta)
        
        weights = weights + this_delta

        if i % 100 == 0:
            error = softmax_test.train_error(weights)
            print("Train error: %.10f" % error)
            train_progress.append(error)

    print("Done iterations!")
    print("Train error: %d", softmax_test.train_error(weights))
    print("Test error: %d", softmax_test.test_error(weights))
    # pdb.set_trace()
    # import sklearn.metrics.pairwise as smp
    # cs = smp.cosine_similarity(summed_deltas)
    return weights


# amazon: 50 classes, 10000 features
# mnist: 10 classes, 784 features
# kdd: 23 classes, 41 features
if __name__ == "__main__":
    argv = sys.argv[1:]

    dataset = argv[0]
    iterations = int(argv[1])

    if (dataset == "mnist"):
        numClasses = 10
        numFeatures = 784
    elif (dataset == "kddcup"):
        numClasses = 23
        numFeatures = 41
    elif (dataset == "amazon"):
        numClasses = 50
        numFeatures = 10000
    else:
        print("Dataset " + dataset + " not found. Available datasets: mnist kddcup amazon")

    numParams = numClasses * numFeatures
    dataPath = dataset + "/" + dataset

    full_model = softmax_model_obj.SoftMaxModel(dataPath + "_train", numClasses)
    Xtest, ytest = full_model.get_data()

    models = []

    for i in range(numClasses):
        # Try a little more IID
        models.append(dataPath + str(i))# + str((i + 1) % 10) + str((i
        # + 2) % 10))

    for attack in argv[2:]:
        attack_delim = attack.split("_")
        sybil_set_size = attack_delim[0]
        from_class = attack_delim[1]
        to_class = attack_delim[2]
        for i in range(int(sybil_set_size)):
            models.append(dataPath + "_bad_" + from_class + "_" + to_class)

    softmax_test = softmax_model_test.SoftMaxModelTest(dataset, numClasses, numFeatures)
    # Hard code poisoners in a 2_x_x attack
    eval_data = np.ones((10, 5))
    for eval_i in range(10):
        topk_prop = 0.1 + eval_i*.1

        weights = non_iid(models, numClasses, numParams, softmax_test, topk_prop, iterations, int(sybil_set_size), ideal_attack=False, poisoner_indices=[10,11])

        for attack in argv[2:]:
            attack_delim = attack.split("_")
            from_class = attack_delim[1]
            to_class = attack_delim[2]
            score = poisoning_compare.eval(Xtest, ytest, weights, int(from_class), int(to_class), numClasses, numFeatures)
            eval_data[eval_i] = score

    np.savetxt('hard_topk_eval_data.csv', eval_data, '%.5f', delimiter=",")
    # # Sandbox: difference between ideal bad model and global model
    # compare = False
    # if compare:
    #     bad_weights = basic_conv(dataPath + "_bad_ideal_" + from_class + "_" +
    #        to_class, numParams, softmax_test)
    #     poisoning_compare.eval(Xtest, ytest, bad_weights, int(from_class),
    #         int(to_class), numClasses, numFeatures)

    #     diff = np.reshape(bad_weights - weights, (numClasses, numFeatures))
    #     abs_diff = np.reshape(np.abs(bad_weights - weights), (numClasses,
    #        numFeatures))

    pdb.set_trace()
