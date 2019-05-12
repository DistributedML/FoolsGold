import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdb

fig, ax = plt.subplots(figsize=(10, 5))

df = pd.read_csv("batches.csv", header=None)
data = df.values

plt.plot(data[0], data[1], color="black", label="MNIST", lw=5)
plt.plot(data[0], data[2], color="red", label="KDDCup", lw=5)
plt.plot(data[0], data[3], color="orange", label="Amazon", lw=5)

plt.legend(loc='center right', ncol=1, fontsize=18)

plt.xlabel("Batch Size", fontsize=22)
plt.ylabel("Attack Rate", fontsize=22)

axes = plt.gca()
axes.set_ylim([0, 1])

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.setp(ax.get_xticklabels(), fontsize=18)
plt.setp(ax.get_yticklabels(), fontsize=18)

fig.savefig("fig4_batches.pdf", bbox_inches='tight')

plt.show()

pdb.set_trace()
