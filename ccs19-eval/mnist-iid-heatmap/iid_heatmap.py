import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import re
import pdb
import pandas as pd

def plot():

    toplot = np.zeros((5, 5))
    props = [0, 25, 50, 75, 100]

    for i in np.arange(5):
        for j in np.arange(5):
            df = pd.read_csv("mnist_iid_h" + str(props[i]) + "_b" + str(props[j]) + ".csv", header=None)
            toplot[i,j] = df[4][0]

    ax = sns.heatmap(toplot, 
        linewidth=0.5, annot=True, fmt=".3f",
        center=0, cmap='Greys')

    plt.xlabel("Honest IID", fontsize=18)
    plt.ylabel("Bad IID", fontsize=18)

    plt.tight_layout(pad=0.1)
    plt.show()


def collect():

    x = 10
    y = 10
    dataset = 'mnist'
    iterations = 3000
    grid_data = []

    for i in range(x):
        row = []
        for j in range(y):
            if i == j:
                row.append(1)
            else:
                filename = 'autologs/play_1/' + dataset + ' ' + str(iterations) + ' 5_' + str(i) + '_' + str(j) + '.log'
                with open(filename, 'r') as logfile:
                    data = logfile.read()
                    #print(data)
                    print(str(i) + str(j))
                    attack_rate_match = re.search('Target Attack Rate.*:\s+([0-9]*.[0-9]*)', data)
                    attack_rate = float(attack_rate_match.group(1))
                    row.append(attack_rate)
        grid_data.append(row)

    data = np.array(grid_data)
    np.savetxt("play1.csv", data, delimiter=',')
    return data

if __name__ == "__main__":

    plot()
    pdb.set_trace()
