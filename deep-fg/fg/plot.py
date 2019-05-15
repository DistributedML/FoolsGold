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
    iid_dir = "../save/vgg_iidness/"
    # iid_dir = "../save/squeeze_iidness/"
    iid_file_path = os.path.join(iid_dir, "100-100.pth")
    iid_exp = torch.load(iid_file_path)
    memory = iid_exp['memory']
    plt.imshow(memory, cmap=plt.cm.gray, vmin=0.0, vmax=10)
    yticks = ["Client " + str(i+1) for i in range(10)]
    yticks.extend(["Sybil " + str(i+1) for i in range(5)])
    plt.yticks(ticks=range(15), labels=yticks)
    plt.xlabel("Parameters")
    plt.axes().set_aspect('auto')
    plt.show()
    pdb.set_trace()

if __name__ == "__main__":
    plot_gradients()
    # iid_dir = "../save/vgg_iidness/"
    # # iid_dir = "../save/squeeze_iidness/"
    # grid_path = os.path.join(iid_dir, "vgg_grid.npy")
    # # grid_path = os.path.join(iid_dir, "fed_grid.npy")
    # grid = np.load(grid_path)
    # np.savetxt("vgg-heatmap.csv", grid, delimiter=",")
    # pdb.set_trace()