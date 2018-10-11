from __future__ import division
from numpy.linalg import norm
import matplotlib.pyplot as plt
import sklearn.metrics.pairwise as smp

import model_aggregator
import softmax_model_test
import softmax_model_obj
import poisoning_compare
import math
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

def rescale(x, a, b):
    minNum = np.min(x)
    maxNum = np.max(x)
    return (b - a)*(x - minNum) / (maxNum - minNum) + a 

def cos(vecA, vecB):
    return np.dot(vecA, vecB)/(np.linalg.norm(vecA) * np.linalg.norm(vecB))

def getOrthogonalNoise(numSybils, numParams):
    numOrthoBasis = int(math.ceil(numSybils / 2.0))

    q, r = np.linalg.qr(np.random.rand(numParams, numOrthoBasis))
    q = q.T
    
    noiseGrad = np.zeros((numSybils, numParams))
    noiseGrad[0:numOrthoBasis] = q
    for i in range(numSybils - numOrthoBasis):
        noiseGrad[numOrthoBasis + i] = - q[i]
    
    return noiseGrad

def rescaleOrthogonalNoise(noise, deltas):
    # maximum norm of organic gradients
    maxNorm = np.max(np.linalg.norm(deltas, axis=1))
    n,d = noise.shape

    for i in range(n):
        noise[i] = noise[i] / (np.linalg.norm(noise[i]) / maxNorm)

    return noise


def getOrthogonalNoise2(numSybils, numParams):
    
    numOrthoBasis = numSybils
    q, r = np.linalg.qr(np.random.rand(numParams, numOrthoBasis))

    return q.T

# Variant of non_iid, where noise is added to poisoner_indices
def non_iid(max_similarity, Xtest, ytest, from_class, to_class, model_names, numClasses, numParams, softmax_test, topk_prop, iterations=3000, numSybils=2,
    ideal_attack=False, poisoner_indices = []):
    numFeatures = int(numParams/numClasses)
    batch_size = 50
    topk = int(numParams / 2)

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

    #### Cosine similarity for adversaries ####
    sybil_noise = getOrthogonalNoise(numSybils, numParams)
    Attack_rates = []
    Attack_rates_iter = []
    numPoisonContribution = 0.0
    for i in xrange(iterations):
        # if i % 10 == 0 and max_similarity < 0.2:
        #     score = poisoning_compare.eval(Xtest, ytest, weights, int(from_class), int(to_class), numClasses, numFeatures, False)
        #     Attack_rates.append(score[4])
        #     Attack_rates_iter.append(i)
        #     print("Attack rate: " + str(score[4]))
        #     # attack success
        #     if score[4] > 0.8:
        #         pdb.set_trace()
        #         return weights, numPoisonContribution / i

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
        # noisevec = rescale(np.random.rand(numParams), np.min(delta), np.max(delta))
        # delta[poisoner_indices[0], :] = delta[poisoner_indices[0], :] + 10000*noisevec
        # delta[poisoner_indices[1], :] = delta[poisoner_indices[1], :] - 10000*noisevec

        ### Adaptive poisoning !! use even number sybils ###
        adaptive = True
        if adaptive:
            # sybil_deltas = summed_deltas[10:10+numSybils].copy()
            # sybil_deltas = sybil_deltas + delta[10:10+numSybils]
            sybil_cs = smp.cosine_similarity(summed_deltas[numClasses:numClasses+numSybils] + delta[numClasses:numClasses+numSybils]) - np.eye(numSybils)
            sybil_cs = np.max(sybil_cs, axis=0)
            # max_similarity = 1.0

            if np.any(sybil_cs > max_similarity):
                delta[numClasses:numClasses+numSybils] = rescaleOrthogonalNoise(sybil_noise, delta)
            else:
                numPoisonContribution += 1.0
        # delta[10:10+numSybils] = getOrthogonalNoise(numSybils, numParams) 
        # pdb.set_trace()
        # pdb:: np.max(smp.cosine_similarity(delta[10:10+numSybils]) - np.eye(numSybils), axis=1)
        ##########################
        

        # Track the total vector from each individual client
        summed_deltas = summed_deltas + delta
        
        # Use Foolsgold
        this_delta = model_aggregator.foolsgold(delta, summed_deltas, sig_features_idx, i, weights, 1.0, importance=True, importanceHard=False)
        # this_delta = model_aggregator.average(delta)
        
        weights = weights + this_delta

        if i % 100 == 0:
            error = softmax_test.train_error(weights)
            print("Train error: %.10f" % error)
            train_progress.append(error)


    print("Done iterations!")
    print("Train error: %d", softmax_test.train_error(weights))
    print("Test error: %d", softmax_test.test_error(weights))

    return weights, numPoisonContribution / iterations


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
    threshholds = [0.29, 0.28, 0.27, 0.26, 0.25]
    # threshholds = [0.5, 0.25, 0.1, 0.01, 0.001]
    num_trials = 5

    eval_data = np.zeros((num_trials*len(threshholds), 3))

    for sim_i in range(len(threshholds)):
        max_similarity = threshholds[sim_i]
        for eval_i in range(num_trials):
            print("Evaluating " + str(eval_i) + "th iteration of " + str(max_similarity))
            topk_prop = 0.05
            weights, ratio = non_iid(max_similarity, Xtest, ytest, from_class, to_class, models, numClasses, numParams, softmax_test, topk_prop, iterations, int(sybil_set_size), ideal_attack=False, poisoner_indices=[10,11])

            for attack in argv[2:]:
                attack_delim = attack.split("_")
                from_class = attack_delim[1]
                to_class = attack_delim[2]
                score = poisoning_compare.eval(Xtest, ytest, weights, int(from_class), int(to_class), numClasses, numFeatures)
            
            eval_data[num_trials*sim_i + eval_i] = [max_similarity, ratio, score[4]]

    np.savetxt('adaptive_attackrate.csv', eval_data, '%.5f', delimiter=",")
