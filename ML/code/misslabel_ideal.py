import numpy as np
import sys
import os
import pdb
''' Mislabels preprocessed data using train_set
Takes input
    - dataset: mnist amazon kddcup
    - from class: eg 4
    - to class : eg 9
 and saves updated dataset as "bad_[dataset]_[from]_[to].npy"
'''

def main(argv):
    
    dataset = argv[0]
    dir = "data/" + dataset + "/"
    fp = dir + dataset + "_train.npy"
    
    data = np.load(os.path.join('../ML', fp))
    data[:, -1][data[:, -1] == int(argv[1])] = int(argv[2])

    save_file = dir + dataset + "_bad_ideal_" + argv[1] + "_" + argv[2]
    print("Generated : " + save_file)
    np.save(save_file, data)

if __name__ == "__main__":
    main(sys.argv[1:])
