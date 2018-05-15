import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import re
import pdb
import pandas as pd

def plot():

    df = pd.read_csv("fig2_data.csv", header=None)
    toplot = df.values
    ax = sns.heatmap(toplot, mask=np.eye(10), 
        linewidth=0.5, annot=True, fmt=".3f",
        center=0, cmap='Greys')

    plt.xlabel("Target Label", fontsize=18)
    plt.ylabel("Source Label", fontsize=18)

    plt.tight_layout(pad=0.1)
    plt.savefig("fig2_heatmap.pdf")


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
