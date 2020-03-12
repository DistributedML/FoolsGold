from __future__ import division

import pdb
import signal
import sys
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm

import model_aggregator
import utils
import softmax_model_test
import softmax_model_obj
import poisoning_compare

from mnist_cnn_model import MNISTCNNModel
from mnist_dataset import MNISTDataset
from client import Client
import torchvision.transforms as transforms
np.set_printoptions(suppress=True)

# Just a simple sandbox for testing out python code, without using Go.
def debug_signal_handler(signal, frame):
    import pdb
    pdb.set_trace()


signal.signal(signal.SIGINT, debug_signal_handler)


def basic_conv(dataset, num_params, iterations=3000):

    batch_size = 5

    # Global
    model = MNISTCNNModel
    dataset = MNISTDataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]) # For single channel
    # transform = transforms.Compose([transforms.ToTensor()])
    client = Client("mnist", "mnist_train", batch_size, model(), dataset, transform)
    test_client = Client("mnist", "mnist_test", batch_size, model(), dataset, transform)
    print("Start training")

    # weights = np.random.rand(31050) / 10
    weights = np.zeros(41386)
    # client.updateModel(weights)
    train_progress = np.zeros(iterations)
    test_progress = np.zeros(iterations)

    for i in range(iterations):
        deltas = client.getGrad()
        # Need to use simpleStep because of momentum
        client.simpleStep(deltas)
        if i % 100 == 0:
            print("Train loss: %.10f" % client.getLoss())

    print("Done iterations!")
    print("Train error: %d", client.getTrainErr())
    test_client.updateModel(client.getModelWeights())
    print("Test error: %d", test_client.getTrainErr())
    return weights


def non_iid(model_names, numClasses, numParams, iterations=3000,
    ideal_attack=False):

    batch_size = 50
    memory_size = 0

    list_of_models = []

    transform = transforms.Compose([transforms.ToTensor()])
    model = MNISTCNNModel
    dataset = MNISTDataset
    numParams = 41386
    train_client = Client("mnist", "mnist_train", batch_size, model(), dataset, transform)
    test_client = Client("mnist", "mnist_test", batch_size, model(), dataset, transform)
    init_weights = train_client.getModelWeights()

    for dataset_name in model_names:
        list_of_models.append(Client("mnist", dataset_name, batch_size, model(), dataset, transform))
        list_of_models[-1].updateModel(init_weights)
    
    # # Include the model that sends the ideal vector on each iteration
    # if ideal_attack:
    #     list_of_models.append(softmax_model_obj.SoftMaxModelEvil(dataPath +
    #        "_bad_ideal_4_9", numClasses))

    numClients = len(list_of_models)
    model_aggregator.init(numClients, numParams, numClasses)

    print("Start training across " + str(numClients) + " clients.")

    # weights = np.random.rand(numParams) / 100.0
    train_progress = []

    delta_memory = np.zeros((numClients, numParams, memory_size))
    summed_deltas = np.zeros((numClients, numParams))

    for i in range(iterations):

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
            
                delta[k, :] = list_of_models[k].getGrad()

                # normalize delta
                if np.linalg.norm(delta[k, :]) > 1:
                    delta[k, :] = delta[k, :] / np.linalg.norm(delta[k, :])

                delta_memory[k, :, i % memory_size] = delta[k, :]

            # Track the total vector from each individual client
            summed_deltas = np.sum(delta_memory, axis=2)

        else:

            for k in range(len(list_of_models)):

                delta[k, :] = list_of_models[k].getGrad()

                # normalize delta
                if np.linalg.norm(delta[k, :]) > 1:
                    delta[k, :] = delta[k, :] / np.linalg.norm(delta[k, :])

            # Track the total vector from each individual client
            summed_deltas = summed_deltas + delta
        
        ##################################
        # Use FoolsGold or something else
        ##################################
        # Use Foolsgold (can optionally clip gradients via Krum)
        weights = list_of_models[0].getModelWeights()
        this_delta = model_aggregator.foolsgold(delta,
           summed_deltas, sig_features_idx, i, weights, clip=0)
        
        # Mean
        # this_delta = model_aggregator.average(delta)
        
        # Krum
        # this_delta = model_aggregator.krum(delta, clip=1)
        

        # Step in new gradient direction
        for k in range(len(list_of_models)):
            list_of_models[k].simpleStep(this_delta)

        
        if i % 20 == 0:
            loss = 0.0
            for i in range(10):
                loss += list_of_models[i].getLoss()
            print("Average loss is " + str(loss / len(list_of_models)))

    print("Done iterations!")
    train_client.updateModel(weights)
    test_client.updateModel(weights)
    print("Train error: %d", train_client.getTrainErr())
    print("Test error: %d", test_client.getTrainErr()) 

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
    dataPath = dataset

    # full_model = softmax_model_obj.SoftMaxModel(dataPath + "_train", numClasses)
    # Xtest, ytest = full_model.get_data()

    num_executions = 1
    eval_data = np.zeros(num_executions)

    full_model = softmax_model_obj.SoftMaxModel("mnist/mnist_test", numClasses)
    Xtest, ytest = full_model.get_data()

    for run in range(num_executions):

        models = []

        for i in range(numClasses):
            # Try a little more IID
            models.append(dataPath + str(i))

        for attack in argv[2:]:
            attack_delim = attack.split("_")
            sybil_set_size = attack_delim[0]
            from_class = attack_delim[1]
            to_class = attack_delim[2]
            for i in range(int(sybil_set_size)):
                models.append(dataPath + "_bad_" + from_class + "_" + to_class)

        
        weights = non_iid(models, numClasses, numParams, iterations, ideal_attack=False)
        # weights = basic_conv(dataset, numParams, iterations=3000)

        # for attack in argv[2:]:
        #     attack_delim = attack.split("_")
        #     from_class = attack_delim[1]
        #     to_class = attack_delim[2]
        #     score = poisoning_compare.eval(Xtest, ytest, weights, int(from_class), int(to_class), numClasses, numFeatures)
        #     eval_data[run] = score

    
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
