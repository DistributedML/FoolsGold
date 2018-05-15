import numpy as np
import pdb

probs = [20, 40, 60, 80]
total_samples = 5000

other_from = 0
other_to = 8

for p in probs:

    data1 = np.load("mnist1.npy")
    data4 = np.load("mnist" + str(other_from) + ".npy")

    num_1 = p * 50  # percentage
    num_4 = (100 - p) * 50

    Xones = data1[np.random.permutation(data1.shape[0])[0:num_1]]
    Xfors = data4[np.random.permutation(data4.shape[0])[0:num_4]]

    Xones[:, -1] = 7
    Xfors[:, -1] = other_to

    data = np.vstack((Xones, Xfors))

    np.save("mnist_bad_mixed_" + str(other_from) + str(other_to) + "_" + str(p), data)
