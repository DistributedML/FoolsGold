import pandas as pd
import numpy as np

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pdb

fig, ax = plt.subplots(figsize=(10, 5))

data1 = np.zeros((10, 5))
data2 = np.zeros((10, 5))
data3 = np.zeros((10, 5))
data4 = np.zeros((10, 5))
data5 = np.zeros((10, 5))
data6 = np.zeros((10, 5))
data7 = np.zeros((10, 5))

for i in range(5):
	df = pd.read_csv("increasing_mean_" + str(i) + ".csv", header=None)
	data1[:, i] = df.values[:, 4]

for i in range(5):
	df = pd.read_csv("increasing_krum_" + str(i) + ".csv", header=None)
	data2[:, i] = df.values[:, 4]

for i in range(5):
	df = pd.read_csv("increasing_median_" + str(i) + ".csv", header=None)
	data3[:, i] = df.values[:, 4]

for i in range(5):
	df = pd.read_csv("increasing_trimmean10_" + str(i) + ".csv", header=None)
	data4[:, i] = df.values[:, 4]

for i in range(5):
	df = pd.read_csv("increasing_trimmean20_" + str(i) + ".csv", header=None)
	data5[:, i] = df.values[:, 4]

for i in range(5):
	df = pd.read_csv("increasing_fg_" + str(i) + ".csv", header=None)
	data6[:, i] = df.values[:, 4]

for i in range(5):
	df = pd.read_csv("label_fgavg_" + str(i) + ".csv", header=None)
	data7[:, i] = df.values[:, 4]


l1 = mlines.Line2D(np.arange(10), np.mean(data1, axis=1), label="Baseline", marker='*', color='black', markersize=10)
l2 = mlines.Line2D(np.arange(10), np.mean(data2, axis=1), label="Multi-Krum", marker='|', color='red', markersize=10)
l3 = mlines.Line2D(np.arange(10), np.mean(data3, axis=1), label="Median", marker='+', color='red', markersize=10)
l4 = mlines.Line2D(np.arange(10), np.mean(data4, axis=1), label="Trimmed Mean (B = 0.1)", marker='>', color='red', markersize=10)
l5 = mlines.Line2D(np.arange(10), np.mean(data5, axis=1), label="Trimmed Mean (B = 0.2)" , marker='*', color='red', markersize=10)
l6 = mlines.Line2D(np.arange(10), np.mean(data6, axis=1), label="FoolsGold (FED SGD)", marker='|', color='green', markersize=10)
l7 = mlines.Line2D(np.arange(10), np.mean(data7, axis=1), label="FoolsGold (FED AVG)", marker='*', color='green', markersize=10)

ax.add_line(l1)
ax.add_line(l2)
ax.add_line(l3)
ax.add_line(l4)
ax.add_line(l5)
ax.add_line(l6)
ax.add_line(l7)
ax.set_xlim(0, 9.5)

plt.legend(handles=[l1, l2, l3, l4, l5, l6, l7], loc='right', fontsize=22)

plt.xlabel("Number of Poisoners", fontsize=22)
plt.ylabel("Attack Rate", fontsize=22)

axes = plt.gca()
axes.set_ylim([0, 1.05])

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.setp(ax.get_xticklabels(), fontsize=20)
plt.setp(ax.get_yticklabels(), fontsize=20)

fig.tight_layout(pad=0.1)
fig.savefig("fig_baselines.pdf")

plt.show()