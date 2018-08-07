import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pdb

fig, ax = plt.subplots(figsize=(5, 5))

df = pd.read_csv("threshhold_trend.csv", header=None)
data = df.values

plt.scatter(data[:,0], data[:,1], marker='*', color='red', s=50)

plt.xlabel("Poisoning Efficacy", fontsize=20)
plt.ylabel("Similarity Threshold", fontsize=20)

axes = plt.gca()
axes.set_xlim([0, 1.1])
axes.set_ylim([0, 1.1])

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.setp(ax.get_xticklabels(), fontsize=16)
plt.setp(ax.get_yticklabels(), fontsize=16)

fig.tight_layout(pad=0.1)
fig.savefig("fig_perf.pdf")

plt.show()