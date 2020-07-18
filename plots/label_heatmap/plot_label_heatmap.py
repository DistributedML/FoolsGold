import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import re
import pdb
import pandas as pd

def plot():

    df = pd.read_csv("label_heatmap_data.csv", header=None)
    toplot = df.values
    ax = sns.heatmap(toplot, mask=np.eye(10), 
        linewidth=0.5, annot=True, fmt=".3f",
        center=0, cmap='coolwarm')

    plt.xlabel("Target Label", fontsize=18)
    plt.ylabel("Source Label", fontsize=18)

    plt.tight_layout(pad=0.1)
    plt.savefig("fig_label_heatmap.pdf")
    plt.show()

if __name__ == "__main__":

    plot()
