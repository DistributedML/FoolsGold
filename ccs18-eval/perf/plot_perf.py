import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pdb

fig, ax = plt.subplots(figsize=(15, 5))

df = pd.read_csv("perf.csv", header=None)
data = df.values

# plt.plot(data1, color="black", label="Baseline", lw=3)
l1 = mlines.Line2D(data[0], data[1], label="Federated Learning",
	marker='*', color='black', markersize=16)

# plt.plot(data2, color="blue", label="Krum", lw=3)
l2 = mlines.Line2D(data[0], data[2], label="FoolsGold", marker='o', color='orange', markersize=16)

# plt.plot(data1, color="black", label="Baseline", lw=3)
l3 = mlines.Line2D(data[0], data[3], label="Deep Federated Learning ",
	marker='*', color='black', markersize=16)

# plt.plot(data2, color="blue", label="Krum", lw=3)
l4 = mlines.Line2D(data[0], data[4], label="Deep FoolsGold", marker='o',
	color='orange', markersize=16)

ax.add_line(l3)
ax.add_line(l4)
ax.set_xlim(5, 55)

plt.legend(handles=[l4, l3], loc='best', fontsize=24)

plt.xlabel("Number of Clients", fontsize=30)
plt.ylabel("Time (s)", fontsize=30)

axes = plt.gca()
axes.set_ylim([0, 7000])

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.setp(ax.get_xticklabels(), fontsize=24)
plt.setp(ax.get_yticklabels(), fontsize=24)

fig.tight_layout(pad=0.1)
fig.savefig("fig_perf.pdf")

plt.show()