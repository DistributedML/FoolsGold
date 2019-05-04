import matplotlib.pyplot as plt
import pdb
import os
import torch
def plot_gradients(grads):
    plt.imshow(grads)
    plt.axes().set_aspect('auto')
    plt.clf()


if __name__ == "__main__":
    # iid_dir = "../save/iidness_copy/"
    iid_dir = "../save/iidness/"
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