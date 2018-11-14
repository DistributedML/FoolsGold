import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pdb

fig, ax = plt.subplots(figsize=(10, 5))

data1 = np.zeros((10, 5))
data2 = np.zeros((10, 5))
data3 = np.zeros((10, 5))
data4 = np.zeros((10, 5))

for i in range(5):
	df = pd.read_csv("backdoor_fed_" + str(i) + ".csv", header=None)
	data1[:, i] = df.values[:, 2]

for i in range(5):
	df = pd.read_csv("backdoor_krum_" + str(i) + ".csv", header=None)
	data2[:, i] = df.values[:, 2]

for i in range(5):
	df = pd.read_csv("backdoor_fg_" + str(i) + ".csv", header=None)
	data3[:, i] = df.values[:, 2]

for i in range(5):
	df = pd.read_csv("backdoor_fgavg_" + str(i) + ".csv", header=None)
	data4[:, i] = df.values[:, 2]

# plt.plot(data1, color="black", label="Baseline", lw=3)
l1 = mlines.Line2D(np.arange(10), np.mean(data1, axis=1), label="Baseline", marker='*', color='black', markersize=12)

# plt.plot(data2, color="blue", label="Krum", lw=3)
l2 = mlines.Line2D(np.arange(10), np.mean(data2, axis=1), label="Krum", marker='>', color='blue', markersize=12)

# plt.plot(data4, color="orange", label="FoolsGold", lw=3)
l3 = mlines.Line2D(np.arange(10), np.mean(data3, axis=1), label="FoolsGold (FED SGD)", marker='o', color='red', markersize=12)

l4 = mlines.Line2D(np.arange(10), np.mean(data4, axis=1), label="FoolsGold (FED AVG)", marker='s', color='red', markersize=12)

ax.add_line(l1)
ax.add_line(l2)
ax.add_line(l3)
ax.add_line(l4)
ax.set_xlim(0, 9.5)

plt.legend(handles=[l1, l2, l3, l4], loc='right', fontsize=18)

plt.xlabel("Number of Backdoor Poisoners", fontsize=22)
plt.ylabel("Attack Rate", fontsize=22)

axes = plt.gca()
axes.set_ylim([0, 1.05])

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.setp(ax.get_xticklabels(), fontsize=18)
plt.setp(ax.get_yticklabels(), fontsize=18)

fig.tight_layout(pad=0.1)
fig.savefig("fig_backdoor_baselines.pdf")

plt.show()