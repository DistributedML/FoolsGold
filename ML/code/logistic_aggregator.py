from __future__ import division
import numpy as np
import pdb
import falconn
import sklearn.metrics.pairwise as smp

n = 0
d = 0
hit_matrix = np.zeros(1)
it = 0


def init(num_clients, num_features):

    global d
    d = num_features

    global n
    n = num_clients

    # global hit_matrix
    # hit_matrix = np.zeros((n, n))


'''
Returns the pairwise cosine similarity of client gradients
'''
def get_cos_similarity(full_deltas):
    if True in np.isnan(full_deltas):
        pdb.set_trace()
    return smp.cosine_similarity(full_deltas)


'''
Aggregates history of gradient directions
'''
def cos_aggregate_sum_nolog(full_deltas, sum_deltas, i):
    deltas = np.reshape(full_deltas, (n, d))
    full_grad = np.zeros(d)

    cs = smp.cosine_similarity(sum_deltas) - np.eye(n)
    maxcs = np.max(cs, axis=1)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if maxcs[i] < maxcs[j]:
                cs[i][j] = cs[i][j] * maxcs[i]/maxcs[j]
    wv = 1 - (np.max(cs, axis=1))
    wv[wv > 1] = 1
    wv[wv < 0] = 0


    full_grad += np.dot(deltas.T, wv)

    return full_grad

'''
Aggregates history of gradient directions
'''
def cos_aggregate_sum_norecalib(full_deltas, sum_deltas, i):
    deltas = np.reshape(full_deltas, (n, d))
    full_grad = np.zeros(d)

    cs = smp.cosine_similarity(sum_deltas) - np.eye(n)
    wv = 1 - (np.max(cs, axis=1))
    wv[wv > 1] = 1
    wv[wv < 0] = 0
    # Rescale so that max value is wv
    wv = wv / np.max(wv)
    wv[(wv == 1)] = .99
    wv = (np.log(wv / (1 - wv)) + 0.5)

    wv[(np.isinf(wv) + wv > 1)] = 1

    wv[(wv < 0)] = 0

    full_grad += np.dot(deltas.T, wv)

    return full_grad



'''
Aggregates history of gradient directions
'''
def cos_aggregate_sum(full_deltas, sum_deltas, i):
    deltas = np.reshape(full_deltas, (n, d))
    full_grad = np.zeros(d)

    cs = smp.cosine_similarity(sum_deltas) - np.eye(n)
    maxcs = np.max(cs, axis=1)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if maxcs[i] < maxcs[j]:
                cs[i][j] = cs[i][j] * maxcs[i]/maxcs[j]
    wv = 1 - (np.max(cs, axis=1))
    wv[wv > 1] = 1
    wv[wv < 0] = 0
    # Rescale so that max value is wv
    wv = wv / np.max(wv)
    wv[(wv == 1)] = .99
    wv = (np.log(wv / (1 - wv)) + 0.5)

    wv[(np.isinf(wv) + wv > 1)] = 1

    wv[(wv < 0)] = 0

    full_grad += np.dot(deltas.T, wv)
    return full_grad
'''
Aggregates history of gradient directions
'''
def cos_aggregate_sum_nomem(full_deltas):
    deltas = np.reshape(full_deltas, (n, d))
    full_grad = np.zeros(d)

    cs = smp.cosine_similarity(full_deltas) - np.eye(n)
    maxcs = np.max(cs, axis=1)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if maxcs[i] < maxcs[j]:
                cs[i][j] = cs[i][j] * maxcs[i]/maxcs[j]
    wv = 1 - (np.max(cs, axis=1))
    wv[wv > 1] = 1
    wv[wv < 0] = 0
    # Rescale so that max value is wv
    wv = wv / np.max(wv)
    wv[(wv == 1)] = .99
    wv = (np.log(wv / (1 - wv)) + 0.5)

    wv[(np.isinf(wv) + wv > 1)] = 1

    wv[(wv < 0)] = 0

    full_grad += np.dot(deltas.T, wv)

    return full_grad

'''
Aggregates history of cosine similarities
'''
def cos_aggregate(full_deltas, cs, i):
    if True in np.isnan(full_deltas):
        pdb.set_trace()
    deltas = np.reshape(full_deltas, (n, d))
    full_grad = np.zeros(d)

    ncs = cs - (i+1)*np.eye(n)
    wv = 1 - (np.max(ncs, axis=1) / (i+1))
    wv[wv > 1] = 1
    wv[wv < 0] = 0
    # Rescale so that max value is wv
    wv = wv / np.max(wv)

    wv = (np.log(wv / (1 - wv)) + 0.5)

    wv[(np.isinf(wv) + wv > 1)] = 1

    wv[(wv < 0)] = 0

    full_grad += np.dot(deltas.T, wv)
    return full_grad

def cos_aggregate_nomem(full_deltas, scs):
    deltas = np.reshape(full_deltas, (n, d))
    full_grad = np.zeros(d)

    scs = scs - np.eye(n)
    wv = 1 - (np.max(scs, axis=1))
    wv[wv > 1] = 1
    wv[wv < 0] = 0
    # Rescale so that max value is wv
    wv = wv / np.max(wv)

    wv = (np.log(wv / (1 - wv)) + 0.5)

    wv[(np.isinf(wv) + wv > 1)] = 1

    wv[(wv < 0)] = 0

    full_grad += np.dot(deltas.T, wv)
    return full_grad

def average(full_deltas):

    deltas = np.reshape(full_deltas, (n, d))
    return np.mean(deltas, axis=0)


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
