import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pdb

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(10, 5))
width = 0.5

ticklabels = ["A-1 MNIST", "A-1 KDD", "A-1 Amazon", "A-5 MNIST", "A-5 KDD",
              "A-5 Amazon", "A-2x5 MNIST", "A-2x5 Amazon", "A-5x5 MNIST", "A-5x5 Amazon",
              "A-AllOnOne ", "A-99"]

ticklabels_mnist = ["Baseline No Attack", "FoolsGold No Attack",
            "A-1", "A-5", "A-2x5", "A-5x5"]

is_mnist = [2, 5, 8, 11, 13, 15]

df1 = pd.read_csv("canon_rate.csv", header=None)
data1 = df1.values
toplot = np.mean(data1, axis=1)

# ax1.xticks(np.arange(12) - 0.5, ticklabels, rotation=45)

# plt.xlabel("Client Label", fontsize=18)
# plt.ylabel("Attack Rate", fontsize=18)

# fig.tight_layout(pad=0.1)
# fig.savefig("canon_rate.pdf")

df2 = pd.read_csv("canon_accuracy.csv", header=None)
data2 = df2.values
toplot2 = np.mean(data2, axis=1)

toplot[toplot < 0.01] = 0.005
toplot2[toplot2 < 0.01] = 0.005

plt.subplot(2, 1, 1)
plt.ylim(0, 1)
plt.bar(np.arange(6), toplot[is_mnist], width)
plt.ylabel("Attack Rate", fontsize=18)
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are of

plt.tick_params(
    labelsize=14,
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are of

plt.subplot(2, 1, 2)
plt.ylim(0, 1.05)
plt.tick_params(
    labelsize=14,
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are of

plt.bar(np.arange(6), toplot2[is_mnist], width)
plt.ylabel("Accuracy", fontsize=18)
plt.xticks(np.arange(6), ticklabels_mnist, rotation=20, fontsize=16)

fig.tight_layout(pad=0.1)
fig.savefig("fig_canon_amazon.pdf")
plt.show()

# pdb.set_trace()
