from __future__ import division
from numpy.linalg import norm
import matplotlib.pyplot as plt

import ML_main as foolsgold
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

    full_model = softmax_model_obj.SoftMaxModel(dataPath + "_test", numClasses)
    Xtest, ytest = full_model.get_data()

    # backdoor_model = softmax_model_obj.SoftMaxModel(dataPath + "_backdoor_test", numClasses)
    # Xback, yback = backdoor_model.get_data()

    softmax_test = softmax_model_test.SoftMaxModelTest(dataset, numClasses, numFeatures)
        
    for run in range(5):

        eval_data = np.zeros((10, 3))
        
        for sybil_count in range(10):

            from_class = "b"
            to_class = "7"

            attack_configs = [{ 
                'sybils': sybil_count, 
                'from': from_class, 
                'to': to_class
            }]

            models = foolsgold.setup_clients(dataPath, numClasses, attack_configs)
            weights = foolsgold.non_iid(models, numClasses, numParams, softmax_test, iterations=iterations, ideal_attack=False)

            if from_class == "b":
                backdoor_model = softmax_model_obj.SoftMaxModel(dataPath + "_backdoor_test", numClasses)
                Xback, yback = backdoor_model.get_data()
                score = poisoning_compare.backdoor_eval(Xback, yback, weights, int(to_class), numClasses, numFeatures)
            else:
                score = poisoning_compare.eval(Xtest, ytest, weights, int(from_class), int(to_class), numClasses, numFeatures)
            
            eval_data[sybil_count] = score

        np.savetxt("increasing_back_mean_" + str(run) + ".csv", eval_data, fmt='%.5f', delimiter=',')