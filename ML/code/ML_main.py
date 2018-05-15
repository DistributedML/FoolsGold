from __future__ import division
from numpy.linalg import norm
import matplotlib.pyplot as plt


import logistic_aggregator
import softmax_model
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


def basic_conv():

    dataset = "mnist_train"

    batch_size = 1
    iterations = 4000
    epsilon = 5

    # Global
    numFeatures = softmax_model.init(dataset, epsilon=epsilon)

    print("Start training")

    weights = np.random.rand(numFeatures) / 1000.0

    train_progress = np.zeros(iterations)
    test_progress = np.zeros(iterations)

    for i in xrange(iterations):
        deltas = softmax_model.privateFun(1, weights, batch_size)
        weights = weights + deltas

        if i % 100 == 0:
            print("Train error: %d", softmax_model_test.train_error(weights))
            print("Test error: %d", softmax_model_test.test_error(weights))

    print("Done iterations!")
    print("Train error: %d", softmax_model_test.train_error(weights))
    print("Test error: %d", softmax_model_test.test_error(weights))


def non_iid(model_names, numClasses, numParams, softmax_test, iter=3000):

    batch_size = 50
    iterations = iter
    epsilon = 5

    list_of_models = []

    for dataset in model_names:
        list_of_models.append(softmax_model_obj.SoftMaxModel(dataset, epsilon, numClasses))

    numClients = len(list_of_models)
    logistic_aggregator.init(numClients, numParams)

    print("Start training across " + str(numClients) + " clients.")

    weights = np.random.rand(numParams) / 100.0
    train_progress = []


    #sum yourself
    #sum pairwise
    ds = np.zeros((numClients, numParams))
    #cs = np.zeros((numClients, numClients))
    for i in xrange(iterations):

        total_delta = np.zeros((numClients, numParams))

        for k in range(len(list_of_models)):
            total_delta[k, :] = list_of_models[k].privateFun(1, weights, batch_size)


        initial_distance = np.random.rand()*10
        ds = ds + total_delta
        #scs = logistic_aggregator.get_cos_similarity(total_delta)
        #cs = cs + scs
        # distance, poisoned = logistic_aggregator.search_distance_euc(total_delta, initial_distance, False, [], np.zeros(numClients), 0, scs)
        # delta, dist, nnbs = logistic_aggregator.euclidean_binning_hm(total_delta, distance, logistic_aggregator.get_nnbs_euc_cos, scs)
        #print(distance)
        delta = logistic_aggregator.cos_aggregate_sum(total_delta, ds, i)
        #delta = logistic_aggregator.cos_aggregate_sum_nomem(total_delta)
        weights = weights + delta

        if i % 100 == 0:
            error = softmax_test.train_error(weights)
            print("Train error: %.10f" % error)
            train_progress.append(error)
    #pdb.set_trace()
    print("Done iterations!")
    print("Train error: %d", softmax_test.train_error(weights))
    print("Test error: %d", softmax_test.test_error(weights))
    return weights


# amazon: 50 classes, 10000 features
# mnist: 10 classes, 784 features
# kdd: 23 classes, 41 features
if __name__ == "__main__":
    argv = sys.argv[1:]

    dataset = argv[0]
    iter = int(argv[1])

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

    full_model = softmax_model_obj.SoftMaxModel(dataPath + "_train", 1, numClasses)
    Xtest, ytest = full_model.get_data()

    models = []

    for i in range(numClasses):
        models.append(dataPath + str(i))

    for attack in argv[2:]:
        attack_delim = attack.split("_")
        sybil_set_size = attack_delim[0]
        from_class = attack_delim[1]
        to_class = attack_delim[2]
        for i in range(int(sybil_set_size)):
            models.append(dataPath + "_bad_" + from_class + "_" + to_class)

    softmax_test = softmax_model_test.SoftMaxModelTest(dataset, numClasses, numFeatures)
    weights = non_iid(models, numClasses, numParams, softmax_test, iter)

    for attack in argv[2:]:
        attack_delim = attack.split("_")
        from_class = attack_delim[1]
        to_class = attack_delim[2]
        score = poisoning_compare.eval(Xtest, ytest, weights, int(from_class), int(to_class), numClasses, numFeatures)
    # pdb.set_trace()
