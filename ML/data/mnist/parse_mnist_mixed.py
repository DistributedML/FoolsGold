import numpy as np
import pdb 

probs = [20, 40, 50, 60, 80]
total_samples = 5000

for i in range(5):

    for p in probs:

        data1 = np.load("mnist1.npy")
        data4 = np.load("mnist4.npy")

        num_1 = p * 50  # percentage
        num_4 = (100 - p) * 50

        Xones = data1[np.random.permutation(data1.shape[0])[0:num_1]]
        Xfors = data4[np.random.permutation(data4.shape[0])[0:num_4]]

        Xones[:, -1] = 7
        Xfors[:, -1] = 9

        data = np.vstack((Xones, Xfors))

        np.save("mnist_bad_mixed_" + str(p) + "_" + str(i), data)
