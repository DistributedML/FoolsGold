import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pdb


def main():

    N = 6
    width = 0.25
    coreidx = np.array([0, 3, 6, 9, 12, 15])

    fig, ax = plt.subplots(figsize=(12, 6))

    ticklabels = ["Baseline TestErr", "Baseline AttackRate", "Baseline Preserved"
                  "NoMem TestErr", "NoMem AttackRate", "NoMem Preserved",
                  "Baseline TestErr", "Baseline AttackRate", "Baseline Preserved",
                  "No RW TestErr", "No RW AttackRate", "No RW Preserved",
                  "Baseline TestErr", "Baseline AttackRate", "Baseline Preserved",
                  "NoLogit TestErr", "NoLogit AttackRate", "NoLogit Preserved"]

    ticklabels = ["MNIST Baseline", "MNIST No Memory", "KDD Baseline",
                  "KDD No Pardoning", "MNIST Baseline 99%", "MNIST No Logit"]

    df1 = pd.read_csv("module.csv", header=None)
    data1 = df1.values

    toplot = np.mean(data1, axis=1)

    toplot[toplot < 0.0001] = 0.0001

    p1 = ax.bar(np.arange(6), toplot[coreidx], width, hatch='/')
    p2 = ax.bar(np.arange(6) + width, toplot[coreidx + 1], width, hatch='\\')
    p3 = ax.bar(np.arange(6) + 2 * width, 1 - toplot[coreidx + 2], width, hatch='.')
    ax.set_xticks(np.arange(6) + width)
    ax.set_xticklabels(ticklabels, rotation=20, fontsize=20)

    yticks = np.array([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels(yticks, fontsize=20)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # create a list to collect the plt.patches data
    totals = []

    # find the values and append to list
    for i in ax.patches:
        totals.append(i.get_height())

    # set individual bar lables using above list
    total = sum(totals)
    str(round((i.get_height() / total), 2))
    # set individual bar lables using above list
    for i in ax.patches:
        # get_x pulls left or right; get_height pushes up or down
        height = str(i.get_height() * 100)[0:4]

        if i.get_height() < 0.001:
            height = "  0"

        ax.text(i.get_x(), i.get_height() + .03, height, fontsize=14, color='black')

    ##############################

    plt.ylim(0, 1.25)

    plt.ylabel('%', fontsize=20)

    ax.legend((p1[0], p2[0], p3[0]),
              ('Test Error', 'Attack Rate', 'Target Class Test Error'),
              loc='upper left', ncol=3, fontsize=20)

    fig.tight_layout(pad=0.1)
    fig.savefig("fig_modules.pdf")

    plt.show()


def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                '%d' % int(height),
                ha='center', va='bottom')


if __name__ == "__main__":
    main()
