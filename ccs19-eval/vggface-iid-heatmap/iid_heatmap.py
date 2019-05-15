import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import re
import pdb
import pandas as pd

def plot():

    toplot = np.zeros((5, 5))
    props = [0, 25, 50, 75, 100]

    df = pd.read_csv("squeeze-heatmap.csv", header=None)
    toplot = df.values

    sns.set(font_scale=1.4)

    ax = sns.heatmap(toplot, 
        linewidth=0.5, 
        annot=True,
        annot_kws={"size": 14},
        fmt=".3f",
        center=0, 
        cmap='Greys',
        xticklabels=props,
        yticklabels=props,
        vmin=0,
        vmax=1
        )

    plt.xlabel("Honest Shared Data Proportion", fontsize=18)
    plt.ylabel("Sybil Shared Data Proportion", fontsize=18)

    plt.tight_layout(pad=0.1)
    plt.savefig("vgg_iid_heatmap.pdf")
    plt.show()

if __name__ == "__main__":

    plot()
