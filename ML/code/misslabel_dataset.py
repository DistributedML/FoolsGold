import numpy as np
import sys
import os
import pdb 

''' Mislabels preprocessed data
Takes input
    - dataset: mnist amazon kddcup
    - from class: eg 4
    - to class : eg 9
 and saves updated dataset as "bad_[dataset]_[from]_[to].npy"
'''
def main(argv):

    if argv[0] == "backdoor":

        dataset = argv[1]
        target = argv[2]
        filename = dataset + "_uniform_" + str(argv[2])
        dir = "data/" + dataset + "/"

        data = np.load(os.path.join('../ML', dir + filename + '.npy'))
        
        # Set the target label
        data[:, -1] = int(argv[2])
        
        # Set pixel #783 to white
        data[:,783] = np.max(data[:,0:784])

        save_file = dir + dataset + "_backdoor_" + argv[2]
        
        print("Generated : " + save_file + " of size " + str(data.shape))
        np.save(save_file, data)    

        testdata = np.load(os.path.join('../ML', dir + 'mnist_test.npy'))
        
        # For now, let's insert the backdoor into 20% of the test data
        testdata[:2000,783] = np.max(testdata[:2000,:784])

        test_save_file = dir + dataset + "_backdoor_test"
        
        print("Generated : " + test_save_file + " of size " + str(testdata.shape))
        np.save(test_save_file, testdata)    

    elif argv[0] == "untargeted":

        dataset = argv[1]
        target = argv[2]
        filename = dataset + "_uniform_" + str(argv[2])
        dir = "data/" + dataset + "/"

        data = np.load(os.path.join('../ML', dir + filename + '.npy'))
        
        # Perturb each label by 1
        data[:, -1] = (data[:, -1] + 1) % 10

        save_file = dir + dataset + "_untargeted_" + argv[2]
        
        print("Generated : " + save_file + " of size " + str(data.shape))
        np.save(save_file, data)

    else:

        dataset = argv[0]
        filename = dataset + argv[1]
        dir = "data/" + dataset + "/"

        data = np.load(os.path.join('../ML', dir + filename + '.npy'))
        data[:, -1] += int(argv[2]) - int(argv[1])
        
        save_file = dir + dataset + "_bad_" + argv[1] + "_" + argv[2]
        print("Generated : " + save_file)
        np.save(save_file, data)

if __name__ == "__main__":
    
    main(sys.argv[1:])
