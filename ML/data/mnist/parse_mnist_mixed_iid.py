import numpy as np
import pdb 

shared_proportion = [0, 25, 50, 75, 100]
total_samples = 5000

for p in shared_proportion:

    for k in range(10):

        base_data = np.load("mnist" + str(k) + ".npy")
        full_data = np.load("mnist_train.npy")

        num_base = (100 - p) * 50
        num_uniform = p * 50

        Xbase = base_data[np.random.permutation(base_data.shape[0])[0:num_base]]
        Xunif = full_data[np.random.permutation(full_data.shape[0])[0:num_uniform]]

        data = np.vstack((Xbase, Xunif))

        print("slice " + str(k) + " is shape " + str(data.shape))
        np.save("mnist" + str(k) + "_mixed" + str(p), data)

        # Create attack 1-7 dataset
        if k == 1:
            ones_idx = np.where(data[:,-1] == 1)[0]
            data[ones_idx, -1] = 7
            np.save("mnist_bad_1_7_mixed" + str(p), data)

            pdb.set_trace()
