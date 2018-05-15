import matplotlib.pyplot as plt
import numpy as np
import inversion_compare

import pdb

if __name__ == "__main__":

    fig, ax = plt.subplots()

    dataw = np.loadtxt("inversion5RUNSWITH.csv", delimiter=',')
    datawo = np.loadtxt("inversion5RUNSWITHOUT.csv", delimiter=',')

    # epsilon from 0.5 to 5
    idx = 0.5 + (np.arange(10) / 2.0)
    plt.errorbar(idx, np.median(dataw, axis=0), yerr=np.std(dataw, axis=0), 
        label="with attacker gradient", lw=5)
    plt.errorbar(idx + 0.05, np.median(datawo, axis=0), yerr=np.std(datawo, axis=0), 
        label="without attacker gradient", lw=5)

    plt.ylabel("Reconstuction Error", fontsize=22)
    plt.xlabel(r'privacy parameter $\varepsilon$', fontsize=22)

    plt.legend(loc='upper right', fontsize=18)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.setp(ax.get_xticklabels(), fontsize=18)
    plt.setp(ax.get_yticklabels(), fontsize=18)

    plt.tight_layout()
    plt.show()
