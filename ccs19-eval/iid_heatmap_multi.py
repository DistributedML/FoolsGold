import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import re
import pdb
import pandas as pd

def plot():

    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12, 5))
    sns.set({'font.size': 12, 'axes.labelsize': 12, 'legend.fontsize': 14.0, 
    'axes.titlesize': 12, 'xtick.labelsize': 14, 'ytick.labelsize': 14})
    cmap = 'coolwarm'

    plt.subplot(1, 2, 1)
    toplot = np.zeros((5, 5))
    props = [0, 25, 50, 75, 100]

    for i in np.arange(5):
        for j in np.arange(5):
            df = pd.read_csv("mnist-iid-heatmap/mnist_iid_h" + str(props[i]) + "_b" + str(props[j]) + ".csv", header=None)
            toplot[i,j] = df[4][0]

    ax = sns.heatmap(toplot, 
        linewidth=1, 
        annot=True,
        fmt=".3f",
        center=0, 
        cmap=cmap,
        xticklabels=props,
        yticklabels=props,
        vmin=0,
        vmax=1,
        cbar=False
        )

    plt.xlabel("Honest MNIST Shared Data Proportion", fontsize=18)
    plt.ylabel("Sybil Shared Data Proportion", fontsize=18)

    plt.subplot(1, 2, 2)
    toplot = np.zeros((5, 5))

    df = pd.read_csv("vggface-iid-heatmap/squeeze-heatmap.csv", header=None)
    toplot = df.values

    ax = sns.heatmap(toplot, 
        linewidth=1, 
        annot=True,
        fmt=".3f",
        center=0, 
        cmap=cmap,
        xticklabels=props,
        yticklabels=[],
        vmin=0,
        vmax=1
        )

    plt.xlabel("Honest VGGFace2 Shared Data Proportion", fontsize=18)
    plt.ylabel("", fontsize=4)

    plt.tight_layout(pad=0.1)
    plt.savefig("multi_iid_heatmap.pdf")
    plt.show()

if __name__ == "__main__":

    plot()
