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


def non_iid(model_names, numClasses, numParams, softmax_test, iterations=3000,
    ideal_attack=False):

    batch_size = 50
    memory_size = 0

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

    delta_memory = np.zeros((numClients, numParams, memory_size))
    summed_deltas = np.zeros((numClients, numParams))

    for i in xrange(iterations):

        delta = np.zeros((numClients, numParams))

        ##################################
        # Use significant features filter or not
        ##################################
        topk = int(numParams / 2)
        
        # Significant features filter, the top k biggest weights
        # sig_features_idx = np.argpartition(weights, -topk)[-topk:]
        sig_features_idx = np.arange(numParams)

        ##################################
        # Use annealing strategy or not
        ##################################
        if memory_size > 0:

            for k in range(len(list_of_models)):
            
                delta[k, :] = list_of_models[k].privateFun(1, weights,
                   batch_size)

                # normalize delta
                if np.linalg.norm(delta[k, :]) > 1:
                    delta[k, :] = delta[k, :] / np.linalg.norm(delta[k, :])

                delta_memory[k, :, i % memory_size] = delta[k, :]

            # Track the total vector from each individual client
            summed_deltas = np.sum(delta_memory, axis=2)

        else:

            for k in range(len(list_of_models)):

                delta[k, :] = list_of_models[k].privateFun(1, weights, batch_size)

                # normalize delta
                if np.linalg.norm(delta[k, :]) > 1:
                    delta[k, :] = delta[k, :] / np.linalg.norm(delta[k, :])

            # Track the total vector from each individual client
            summed_deltas = summed_deltas + delta
        
        ##################################
        # Use FoolsGold or something else
        ##################################

        # Use Foolsgold (can optionally clip gradients via Krum)
        this_delta = model_aggregator.foolsgold(delta,
           summed_deltas, sig_features_idx, i, weights, clip=0)
        
        # Krum
        # this_delta = model_aggregator.krum(delta, clip=1)
        
        weights = weights + this_delta

        if i % 100 == 0:
            error = softmax_test.train_error(weights)
            print("Train error: %.10f" % error)
            train_progress.append(error)

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

    for run in np.arange(1, 6):

        all_scores = np.zeros((11, 5))
        row = 0

        # for kdd
        # toiter = np.arange(1, 11)

        # for kdd
        toiter = np.concatenate((np.arange(1, 2), (np.arange(0.1, 1,
            0.1) * 23).astype(int))) 

        # for amazone
        # toiter = np.concatenate((np.arange(1, 2), np.arange(5, 55,
        # 5)))

        for ncp in toiter:

            models = []

            ##################################
            # Add the models; can try a little more IID
            ##################################
            for k in range(numClasses):
                
                if ncp != 23:
                
                    datasuf = ""
                    for i in range(ncp):
                        datasuf += "_" + str((k + i) % numClasses)

                else:

                    datasuf = "_train"

                print("Appending " + datasuf)
                models.append(dataPath + datasuf)

            for attack in argv[2:]:
                attack_delim = attack.split("_")
                sybil_set_size = attack_delim[0]
                from_class = attack_delim[1]
                to_class = attack_delim[2]
                for i in range(int(sybil_set_size)):
                    models.append(dataPath + "_bad_" + from_class + "_" + to_class)

            softmax_test = softmax_model_test.SoftMaxModelTest(dataset, numClasses, numFeatures)
            weights = non_iid(models, numClasses, numParams, softmax_test, iterations,
                ideal_attack=False)

            for attack in argv[2:]:
                attack_delim = attack.split("_")
                from_class = attack_delim[1]
                to_class = attack_delim[2]
                score = poisoning_compare.eval(Xtest, ytest, weights, int(from_class), int(to_class), numClasses, numFeatures)
                print ' '.join(format(f, '.5f') for f in score)
                all_scores[row] = score

            row += 1

        np.savetxt("kddiid" + str(run) + ".csv", all_scores,
           fmt='%.5f',
           delimiter=',')

    # Sandbox: difference between ideal bad model and global model
    compare = False
    if compare:
        bad_weights = basic_conv(dataPath + "_bad_ideal_" + from_class + "_" +
           to_class, numParams, softmax_test)
        poisoning_compare.eval(Xtest, ytest, bad_weights, int(from_class),
            int(to_class), numClasses, numFeatures)

        diff = np.reshape(bad_weights - weights, (numClasses, numFeatures))
        abs_diff = np.reshape(np.abs(bad_weights - weights), (numClasses,
           numFeatures))

    pdb.set_trace()
