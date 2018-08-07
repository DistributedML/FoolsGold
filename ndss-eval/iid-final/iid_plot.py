import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pdb

fig, ax = plt.subplots(figsize=(10, 5))

mnist_full_df = np.zeros((11, 5, 5))
amazon_full_df = np.zeros((11, 5, 5))
kdd_full_df = np.zeros((11, 5, 5))

for run in range(1, 6):
	
	df = pd.read_csv("mnistiid" + str(run) + ".csv",
		header=None)
	
	mnist_full_df[:, :, run - 1] = df.values

	df = pd.read_csv("kddiid" + str(run) + ".csv",
		header=None)
	
	kdd_full_df[:, :, run - 1] = df.values

	df = pd.read_csv("amazoniid" + str(run) + ".csv",
		header=None)

	amazon_full_df[:, :, run - 1] = df.values

# Take the mean across 5 runs
# plot_df = np.mean(mnist_full_df, axis=2)

# plt.plot(data1, color="black", label="Baseline", lw=3)
l1 = mlines.Line2D(np.arange(0, 110, 10), np.mean(mnist_full_df,
	axis=2)[:, 0], label="MNIST", marker='*', color='black',
	markersize=16) 

l2 = mlines.Line2D(np.arange(0, 110, 10), np.mean(kdd_full_df,
	axis=2)[:, 0], label="KDDCup", marker='*', color='green',
	markersize=16) 

l3 = mlines.Line2D(np.arange(0, 110, 10), np.mean(amazon_full_df,
	axis=2)[:, 0], label="Amazon", marker='*', color='orange',
	markersize=16) 

ax.add_line(l1)
ax.add_line(l2)
ax.add_line(l3)
ax.set_xlim(-2, 105)

plt.legend(handles=[l1, l2, l3], loc='right', fontsize=18)

plt.xlabel("% of classes per client", fontsize=22)
plt.ylabel("Training Accuracy", fontsize=22)

axes = plt.gca()
axes.set_ylim([0, 1.1])

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.setp(ax.get_xticklabels(), fontsize=18)
plt.setp(ax.get_yticklabels(), fontsize=18)

fig.tight_layout(pad=0.1)
fig.savefig("moreiid.pdf")

plt.show()