import numpy as np

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt
import pandas as pd
import pdb
from matplotlib.lines import Line2D

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(20, 3))
width = 0.5

########################################
## MNIST
########################################

ticklabels_mnist = ["FL-NA", "FG-NA",
"A-1", "A-5", "A-5x5", "A-99"]
is_mnist = [0, 3, 6, 9, 14, 17]

df1 = pd.read_csv("canon_rate.csv", header=None)
data1 = df1.values
toplot = np.mean(data1, axis=1)

df2 = pd.read_csv("canon_accuracy.csv", header=None)
data2 = df2.values
toplot2 = np.mean(data2, axis=1)

toplot[toplot < 0.01] = 0.001
toplot2[toplot2 < 0.01] = 0.001

plt.subplot(1, 4, 1)
plt.ylim(0, 1.35)

# Add markers to plots
plt.plot(np.arange(2, 6), toplot[is_mnist[2:]], marker='D', markersize=10,
   color='r', linestyle="")
plt.bar(np.arange(6), toplot2[is_mnist], width)
plt.ylabel("Rate", fontsize=18)
plt.xlabel("MNIST", fontsize=18)
plt.xticks(np.arange(6), ticklabels_mnist, rotation=25, fontsize=12)
plt.tick_params(
    labelsize=14,
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are of

########################################
## VGG
########################################
ticklabels_vgg = ["FL-NA", "FG-NA",
"A-1", "A-5", "A-5x5"]

df1 = pd.read_csv("deep_accuracy.csv", header=None)
toplot = df1.values

plt.subplot(1, 4, 2)
plt.ylim(0, 1.35)

# Add markers to plots
plt.plot(np.arange(2,5), toplot[1,2:], marker='D', markersize=10,
   color='r', linestyle="")
plt.bar(np.arange(5), toplot[0,:], width)

plt.yticks([])
#plt.ylabel("Accuracy", fontsize=18)

plt.xlabel("VGGFace2", fontsize=18)
plt.xticks(np.arange(5), ticklabels_vgg, rotation=25, fontsize=12)
plt.tick_params(
    labelsize=14,
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are of
#fig.tight_layout(pad=0.5)

########################################
## KDD
########################################

ticklabels_kdd = ["FL-NA", "FG-NA", "A-1", "A-5",
"A-OnOne"]
is_kdd = [1, 4, 7, 10, 16]

df1 = pd.read_csv("canon_rate.csv", header=None)
data1 = df1.values
toplot = np.mean(data1, axis=1)

df2 = pd.read_csv("canon_accuracy.csv", header=None)
data2 = df2.values
toplot2 = np.mean(data2, axis=1)

toplot[toplot < 0.01] = 0.001
toplot2[toplot2 < 0.01] = 0.001

plt.subplot(1, 4, 3)
plt.ylim(0, 1.35)
plt.tick_params(
    labelsize=14,
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are of

# Add markers to plots
plt.plot(np.arange(2, 5), toplot[is_kdd[2:]], marker='D', markersize=10,
   color='r', linestyle="")
plt.bar(np.arange(5), toplot2[is_kdd], width)

plt.yticks([])
#plt.ylabel("Accuracy", fontsize=18)

plt.xlabel("KDD", fontsize=18)
plt.xticks(np.arange(5), ticklabels_kdd, rotation=25, fontsize=12)
#fig.tight_layout(pad=0.5)

########################################
## Amazon
########################################
ticklabels_amazon = ["FL-NA", "FG-NA",
            "A-1", "A-5", "A-5x5"]

is_amazon = [2, 5, 8, 11, 15]

df1 = pd.read_csv("canon_rate.csv", header=None)
data1 = df1.values
toplot = np.mean(data1, axis=1)

df2 = pd.read_csv("canon_accuracy.csv", header=None)
data2 = df2.values
toplot2 = np.mean(data2, axis=1)

toplot[toplot < 0.01] = 0.005
toplot2[toplot2 < 0.01] = 0.005

plt.subplot(1, 4, 4)
plt.ylim(0, 1.35)
plt.tick_params(
    labelsize=14,
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are of

# Add markers to plots
plt.plot(np.arange(2, 5), toplot[is_amazon[2:]], marker='D', markersize=10,
   color='r', linestyle="")
plt.bar(np.arange(5), toplot2[is_amazon], width)

#plt.ylabel("Accuracy", fontsize=18)
plt.yticks([])

plt.xlabel("Amazon", fontsize=18)
plt.xticks(np.arange(5), ticklabels_amazon, rotation=25, fontsize=12)

# Fake data for legend
custom_lines = [Line2D([0], [0], lw=4),
                Line2D([0], [0], color='r', marker='D', markersize=10, linestyle="", lw=4)]

fig.legend(custom_lines, ['Test Accuracy', 'Attack Rate'],
    loc='upper right',
    bbox_to_anchor=(1, 1.05),
    ncol=2, fontsize=18)

plt.tight_layout(pad=0.1)
fig.savefig("fig_canon_multi.pdf")

plt.show()
