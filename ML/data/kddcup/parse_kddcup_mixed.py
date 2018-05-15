import numpy as np

probs = [20, 40, 60, 80]

for p in probs:

    data = np.load("kddcup0.npy")

    num_to_flip = p * data.shape[0] / 100
    flip_idx = np.random.permutation(data.shape[0])[0:num_to_flip]

    # Only label that proportion as 7s
    data[flip_idx, -1] = 11

    print("Flipped " + str(len(flip_idx)) + " samples.")
    np.save("kddcup_bad_0_11_" + str(p), data)
