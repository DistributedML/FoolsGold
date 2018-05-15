import numpy as np
import sys
import os

''' Mislabels preprocessed data
Takes input
    - dataset: mnist amazon kddcup
    - from class: eg 4
    - to class : eg 9
 and saves updated dataset as "bad_[dataset]_[from]_[to].npy"
'''

def main(argv):

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
