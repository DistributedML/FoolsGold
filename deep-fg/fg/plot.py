import matplotlib.pyplot as plt
import pdb

def plot_gradients(grads):
    plt.imshow(grads)
    plt.axes().set_aspect('auto')
    plt.clf()