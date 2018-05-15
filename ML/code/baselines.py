import numpy as np
import utils
import pdb
from sklearn.linear_model import SGDClassifier
import time

data = utils.load_dataset("credit")
X, y = data['X'], data['y']
Xvalid, yvalid = data['Xvalid'], data['yvalid']

if __name__ == "__main__":

    nn = X.shape[0]

    # Sample n points with replacement from n examples
    bootsample = np.random.choice(nn, 10 * nn)
    Xboot = X[bootsample, :]
    yboot = y[bootsample]

    clf = SGDClassifier(loss="hinge", penalty="l2")

    t_start = time.time()
    clf.fit(Xboot, yboot)
    t_end = time.time()

    print "Took " + str(t_end - t_start) + " time"
