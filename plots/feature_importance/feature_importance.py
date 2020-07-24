import pandas as pd
import numpy as np

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pdb

fig, ax = plt.subplots(figsize=(10, 3.5))

full_df = np.zeros((14, 5, 5))

# Get the hard k results
for run in range(1, 6):
	
	df = pd.read_csv("hard_topk_eval_left_" + str(run) + ".csv",
		header=None)

	full_df[0:4, :, run - 1] = df.values

	df = pd.read_csv("hard_topk_eval_data" + str(run) + ".csv",
		header=None)
	
	full_df[4:, :, run - 1] = df.values

# Take the mean across 5 runs
plot_df = np.mean(full_df, axis=2)

# Get soft results
df = pd.read_csv("soft_topk_eval_data.csv", header=None)

soft_df = df.values
soft_plot_df = np.mean(soft_df, axis=0)

xticks = np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4,
	0.5, 0.6, 0.7, 0.8, 0.9, 1])

# plt.plot(data1, color="black", label="Baseline", lw=3)
l1 = mlines.Line2D(xticks, plot_df[:, 4],
	label="Attack Rate", marker='*', color='red', markersize=16)

l2 = mlines.Line2D(xticks, plot_df[:, 0],
	label="Test Accuracy", marker='s', color='green',
	markersize=16)

l3 = mlines.Line2D(xticks, np.full(14, soft_plot_df[4]),
	label="Soft Attack Rate", color='red',
	linestyle='dashed', markersize=16)

l4 = mlines.Line2D(xticks, np.full(14, soft_plot_df[0]),
	label="Soft Test Accuracy", color='green', 
	linestyle='dashed', markersize=16)

ax.add_line(l1)
ax.add_line(l2)
ax.add_line(l3)
ax.add_line(l4)
ax.set_xlim(-0.05, 1.1)

plt.legend(handles=[l1, l2, l3, l4], loc='lower right', bbox_to_anchor=(1, 0.05), fontsize=18)

plt.xlabel("Proportion of Indicative Features", fontsize=22)
plt.ylabel("%", fontsize=22)

axes = plt.gca()
axes.set_ylim([-0.05, 1.1])

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.setp(ax.get_xticklabels(), fontsize=18)
plt.setp(ax.get_yticklabels(), fontsize=18)

fig.tight_layout(pad=0.1)
fig.savefig("fig_feature_importance.pdf")

plt.show()
