import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pdb

fig, ax = plt.subplots(figsize=(10, 5))

df = pd.read_csv("perf.csv", header=None)
data1 = df.values[:, 3]

df = pd.read_csv("fig1results_krum_niid.csv", header=None)
data2 = df.values[:, 3]

data3 = np.zeros(10)
for i in range(5):
    df = pd.read_csv("fig1results_krum_iid_" + str(i) + ".csv", header=None)
    data3 += df.values[:, 3] / 5

data4 = np.zeros(10)
for i in range(3):
    df = pd.read_csv("fig1results_foolsgold_" + str(i) + ".csv", header=None)
    data4 = df.values[:, 3] / 3

# plt.plot(data1, color="black", label="Baseline", lw=3)
l1 = mlines.Line2D(np.arange(10), data1, label="Baseline", marker='*', color='black', markersize=16)

# plt.plot(data2, color="blue", label="Krum", lw=3)
l2 = mlines.Line2D(np.arange(10), data2, label="Krum", marker='>', color='blue', markersize=16)

# plt.plot(data3, color="green", label="Krum (IID)", lw=3)
l3 = mlines.Line2D(np.arange(10), data3, label="Krum (IID)", marker='s', color='green', markersize=16)

# plt.plot(data4, color="orange", label="FoolsGold", lw=3)
l4 = mlines.Line2D(np.arange(10), data4, label="FoolsGold", marker='o', color='orange', markersize=16)

ax.add_line(l1)
ax.add_line(l2)
ax.add_line(l3)
ax.add_line(l4)
ax.set_xlim(0, 9.5)

plt.legend(handles=[l1, l2, l3, l4], loc='right', fontsize=18)

plt.xlabel("Number of Poisoners", fontsize=22)
plt.ylabel("Attack Rate", fontsize=22)

axes = plt.gca()
axes.set_ylim([0, 1])

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.setp(ax.get_xticklabels(), fontsize=18)
plt.setp(ax.get_yticklabels(), fontsize=18)

fig.tight_layout(pad=0.1)
fig.savefig("fig1_baselines.pdf")

plt.show()