from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
import numpy as np
import sys
import utils
import pdb


def compare(victim_csv):

    # Inverted attacked dataset
    data = utils.load_dataset("credit/creditbad")
    X, y = data['X'], data['y']

    testdata = utils.load_dataset("credit/credittest")
    Xvalid, yvalid = testdata['X'], testdata['y']

    # Train the optimal classifier
    clf = SGDClassifier(loss="hinge", penalty="l2")
    clf.fit(X, y)

    # real_model = np.array(np.loadtxt("victimPosModel.csv", delimiter=','))
    # real_hat = np.sign(np.dot(Xvalid, victim_model))
    real_hat = clf.predict(Xvalid)

    victim_data = np.loadtxt(victim_csv, delimiter=',')

    victim_model = np.array(victim_data)
    victim_hat = np.sign(np.dot(Xvalid, victim_model))

    agreement = np.sum(victim_hat != real_hat) / float(real_hat.shape[0])

    print("Error is %.3f" % agreement)

    return agreement


if __name__ == "__main__":

    victimcsv = sys.argv[1]
    compare(victimcsv)
