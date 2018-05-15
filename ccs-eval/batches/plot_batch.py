import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pdb

fig, ax = plt.subplots(figsize=(8, 5))

df1 = pd.read_csv("mnist_batch.csv", header=None)
data1 = df1.values

df2 = pd.read_csv("kddbatch.csv", header=None)
data2 = df2.values

df3 = pd.read_csv("amazonbatch.csv", header=None)
data3 = df3.values

plt.plot(data1[0], np.mean(data1[1:3], axis=0), color="black", label="MNIST", lw=3)
plt.plot(data2[0], np.mean(data2[1:3], axis=0), color="red", label="KDDCup", lw=3)
plt.plot(data3[0], np.mean(data3[1:3], axis=0), color="orange", label="Amazon", lw=3)

plt.legend(loc='center right', ncol=1, fontsize=18)
plt.ylim(0, 0.05)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.setp(ax.get_xticklabels(), fontsize=18)
plt.setp(ax.get_yticklabels(), fontsize=18)

plt.xlabel("Batch Size", fontsize=22)
plt.ylabel("Attack Rate", fontsize=22)

fig.tight_layout(pad=0.1)
fig.savefig("fig_batch.pdf")
plt.show()
