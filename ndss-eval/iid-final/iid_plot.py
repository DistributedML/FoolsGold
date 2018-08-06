import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pdb

fig, ax = plt.subplots(figsize=(10, 5))

full_df = np.zeros((11, 5, 5))

for run in range(1, 6):
	df = pd.read_csv("mnistiid" + str(run) + ".csv",
		header=None)
	full_df[:, :, run - 1] = df.values

# Take the mean across 5 runs
plot_df = np.mean(full_df, axis=2)

# plt.plot(data1, color="black", label="Baseline", lw=3)
l1 = mlines.Line2D(np.arange(0, 110, 10), plot_df[:, 0],
	label="Baseline", marker='*', color='black', markersize=16)

ax.add_line(l1)
ax.set_xlim(0, 105)

plt.legend(handles=[l1], loc='right', fontsize=18)

plt.xlabel("% of classes per client", fontsize=22)
plt.ylabel("Training Accuracy", fontsize=22)

axes = plt.gca()
axes.set_ylim([0, 1])

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.setp(ax.get_xticklabels(), fontsize=18)
plt.setp(ax.get_yticklabels(), fontsize=18)

fig.tight_layout(pad=0.1)
fig.savefig("moreiid.pdf")

plt.show()