from __future__ import division
import numpy as np
import pdb
import sklearn.metrics.pairwise as smp
import matplotlib.pyplot as plt
import scipy.stats

n = 0
d = 0
n_classes = 0
hit_matrix = np.zeros(1)
it = 0
epsilon = 1e-5

def init(num_clients, num_features, num_classes):

    global d
    d = num_features

    global n
    n = num_clients

    global n_classes
    n_classes = num_classes

'''
Returns the pairwise cosine similarity of client gradients
'''
def get_cos_similarity(full_deltas):
    if True in np.isnan(full_deltas):
        pdb.set_trace()
    return smp.cosine_similarity(full_deltas)



def importanceFeatureMapGlobal(model):
    # aggregate = np.abs(np.sum( np.reshape(model, (10, 784)), axis=0))
    # aggregate = aggregate / np.linalg.norm(aggregate)
    # return np.repeat(aggregate, 10)
    return np.abs(model) / np.sum(np.abs(model))

def importanceFeatureMapLocal(model, topk_prop):
    class_d = int(d / n_classes)

    M = model.copy()
    M = np.reshape(M, (n_classes, class_d))
    
    # #Take abs?
    # M = np.abs(M)

    for i in range(n_classes):
        if (M[i].sum() == 0):
            pdb.set_trace()
        M[i] = np.abs(M[i] - M[i].mean())
        
        M[i] = M[i] / M[i].sum()

        # Top k of 784
        topk = int(class_d * topk_prop)
        sig_features_idx = np.argpartition(M[i], -topk)[0:-topk]
        M[i][sig_features_idx] = 0
    
    return M.flatten()

def importanceFeatureHard(model, topk_prop):
    class_d = int(d / n_classes)

    M = np.reshape(model, (n_classes, class_d))
    importantFeatures = np.ones((n_classes, class_d))
    # Top k of 784
    topk = int(class_d * topk_prop)
    for i in range(n_classes):
        sig_features_idx = np.argpartition(M[i], -topk)[0:-topk]     
        importantFeatures[i][sig_features_idx] = 0
    

    return importantFeatures.flatten()
'''
Aggregates history of gradient directions
'''
def foolsgold(this_delta, summed_deltas, sig_features_idx, iter, model, topk_prop=0, importance=False, importanceHard=False, clip=0):

    # Take all the features of sig_features_idx for each clients
    sd = summed_deltas.copy()
    sig_filtered_deltas = np.take(sd, sig_features_idx, axis=1)

    if importance or importanceHard:
        if importance:
            # smooth version of importance features
            importantFeatures = importanceFeatureMapLocal(model, topk_prop)
        if importanceHard:
            # hard version of important features
            importantFeatures = importanceFeatureHard(model, topk_prop)
        for i in range(n):
            sig_filtered_deltas[i] = np.multiply(sig_filtered_deltas[i], importantFeatures)

    cs = smp.cosine_similarity(sig_filtered_deltas) - np.eye(n)
    # Pardoning: reweight by the max value seen
    maxcs = np.max(cs, axis=1) + epsilon
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if maxcs[i] < maxcs[j]:
                cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]

    wv = 1 - (np.max(cs, axis=1))
    wv[wv > 1] = 1
    wv[wv < 0] = 0

    # Rescale so that max value is wv
    wv = wv / np.max(wv)
    wv[(wv == 1)] = .99
    
    # Logit function
    wv = (np.log((wv / (1 - wv)) + epsilon) + 0.5)
    wv[(np.isinf(wv) + wv > 1)] = 1
    wv[(wv < 0)] = 0
    
    # if iter % 10 == 0 and iter != 0:
    #     print maxcs
    #     print wv

    if clip != 0:

        # Augment onto krum
        scores = get_krum_scores(this_delta, n - clip)
        bad_idx = np.argpartition(scores, n - clip)[(n - clip):n]

        # Filter out the highest krum scores
        wv[bad_idx] = 0

    # Apply the weight vector on this delta
    delta = np.reshape(this_delta, (n, d))

    return np.dot(delta.T, wv)

# Simple element-wise mean
def average(full_deltas):

    deltas = np.reshape(full_deltas, (n, d))
    return np.mean(deltas, axis=0)

# Simple element-wise median
def median(full_deltas):

    deltas = np.reshape(full_deltas, (n, d))
    return np.median(deltas, axis=0)

# Beta is the proportion to trim from the top and bottom.
def trimmed_mean(full_deltas, beta):

    deltas = np.reshape(full_deltas, (n, d))
    return scipy.stats.trim_mean(deltas, beta, axis=0)

# Returns the index of the row that should be used in Krum
def krum(deltas, clip):

    # assume deltas is an array of size group * d
    scores = get_krum_scores(deltas, n - clip)
    good_idx = np.argpartition(scores, n - clip)[:(n - clip)]

    return np.mean(deltas[good_idx], axis=0)


def get_krum_scores(X, groupsize):

    krum_scores = np.zeros(len(X))

    # Calculate distances
    distances = np.sum(X**2, axis=1)[:, None] + np.sum(
        X**2, axis=1)[None] - 2 * np.dot(X, X.T)

    for i in range(len(X)):
        krum_scores[i] = np.sum(np.sort(distances[i])[1:(groupsize - 1)])

    return krum_scores


if __name__ == "__main__":

    good = (np.random.rand(50, 5) - 0.5) * 2
    attackers = np.random.rand(10, 5) + 0.5

    sample = np.vstack((good, attackers))

    lsh_sieve(sample.flatten(), 5, 60)

    pdb.set_trace()
