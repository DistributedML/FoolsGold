import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt
import pdb
import os
import torch
import numpy as np

def plot_gradients(grads):
    plt.imshow(grads)
    plt.axes().set_aspect('auto')
    plt.clf()


def plot_gradients():
    
    iid_exp = torch.load("squeeze/0-0.pth")
    memory = iid_exp['memory']

    plt.imshow(memory, cmap=plt.cm.gray, vmin=0.0, vmax=10)
    yticks = ["Client " + str(i+1) for i in range(10)]
    yticks.extend(["Sybil " + str(i+1) for i in range(5)])
    
    plt.xlabel("Parameters")
    plt.axes().set_aspect('auto')
    plt.show()
    pdb.set_trace()

def multiplot_gradients():

    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    iid_exp = torch.load("squeeze/0-0.pth")
    memory = iid_exp['memory']
    plt.imshow(memory, cmap=plt.cm.gray, vmin=0.0, vmax=10, aspect='auto')
    
    yticks = ["Client " + str(i+1) for i in range(10)]
    yticks.extend(["Sybil " + str(i+1) for i in range(5)])
    plt.xticks(fontsize=14)
    plt.yticks(np.arange(15), yticks, fontsize=14)
    plt.xlabel("Parameters (Non-IID)", fontsize=18)

    plt.subplot(1, 2, 2)
    iid_exp = torch.load("squeeze/100-100.pth")
    memory = iid_exp['memory']
    plt.imshow(memory, cmap=plt.cm.gray, vmin=0.0, vmax=10, aspect='auto')
    plt.xlabel("Parameters (Full-IID)", fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks([])

    fig.tight_layout(pad=0.1)
    fig.savefig("fig_squeeze_grads.pdf")
    plt.show()

if __name__ == "__main__":
    multiplot_gradients()
    