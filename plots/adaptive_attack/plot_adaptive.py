import pandas as pd
import numpy as np

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pdb

fig, ax = plt.subplots(figsize=(5, 5))

df = pd.read_csv("threshhold_trend.csv", header=None)
data = df.values

plt.scatter(data[:,0], 1.0 / data[:,1], marker='*', color='red', s=80)

plt.ylabel("Expected Ratio of Sybils", fontsize=18)
plt.xlabel("Similarity Threshold M", fontsize=18)

axes = plt.gca()
axes.set_xlim([-0.05, 1.1])
axes.set_ylim([0, 30])

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.setp(ax.get_xticklabels(), fontsize=12)
plt.setp(ax.get_yticklabels(), fontsize=12)
plt.axvline(x=0.27, color='k', linestyle='--')

fig.tight_layout(pad=0.1)
fig.savefig("fig_adaptive.pdf")

plt.show()