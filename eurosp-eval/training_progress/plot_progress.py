import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.lines as mlines

fig, ax = plt.subplots(2, 1, figsize=(12, 5))

val_attack = pd.read_csv("train_progress_untargeted_20K.csv", header=None)
val_baseline = pd.read_csv("train_progress_baseline_20K.csv", header=None)
val_adaptive = pd.read_csv("train_progress_adaptive_20K.csv", header=None)
val_slow = pd.read_csv("train_progress_slow_20K.csv", header=None)

l5 = ax[0].plot(val_baseline, label="Baseline", color='blue', lw=3)
l6 = ax[0].plot(val_adaptive, label="Adaptive", color='orange', lw=3)
l7 = ax[0].plot(val_slow, label="Adaptive Slow", color='black', lw=3)
l8 = ax[0].plot(val_attack, label="Attacked", color='red', lw=3)

norm_attack = pd.read_csv("norm_progress_untargeted_20K.csv", header=None)
norm_baseline = pd.read_csv("norm_progress_baseline_20K.csv", header=None)
norm_adaptive = pd.read_csv("norm_progress_adaptive_20K.csv", header=None)
norm_slow = pd.read_csv("norm_progress_slow_20K.csv", header=None)

l1 = ax[1].plot(norm_baseline, label="Baseline", color='blue')
l2 = ax[1].plot(norm_adaptive, label="Adaptive", color='orange')
l3 = ax[1].plot(norm_slow, label="Adaptive Slow", color='black')
l4 = ax[1].plot(norm_attack, label="Attacked", color='red')

ax[0].legend(loc='best', fontsize=16, ncol=4)
ax[0].set_ylabel("Validation Error", fontsize=18)
ax[0].set_ylim(0, 0.4)
ax[1].set_ylabel("L2-Norm", fontsize=18)
ax[1].set_xlabel("FL Iterations", fontsize=18)

ax[0].spines['right'].set_visible(False)
ax[0].spines['top'].set_visible(False)
ax[0].set_xticklabels([])

ax[1].spines['right'].set_visible(False)
ax[1].spines['top'].set_visible(False)

ax[0].set_yticklabels([0, 0.1, 0.2, 0.3, 0.4])
# ax[1].set_yticklabels([0, 0.1, 0.2, 0.3, 0.4])

# X tick hacking
ax[1].set_xticklabels(np.arange(-2500, 22500, 2500))

plt.setp(ax[0].get_xticklabels(), fontsize=18)
plt.setp(ax[0].get_yticklabels(), fontsize=18)
plt.setp(ax[1].get_xticklabels(), fontsize=18)
plt.setp(ax[1].get_yticklabels(), fontsize=18)

fig.tight_layout(pad=0.1)
fig.savefig("fig_training.pdf")

plt.show()