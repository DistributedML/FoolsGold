import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pdb


df1 = pd.read_csv("mnist_batch.csv", header=None)
data1 = df1.values

df2 = pd.read_csv("kdd_batch.csv", header=None)
data2 = df2.values

df3 = pd.read_csv("amazon_batch.csv", header=None)
data3 = df3.values

# plt.plot(data1[0], np.mean(data1[1:3], axis=0), color="black", label="MNIST", lw=3)
# plt.plot(data2[0], np.mean(data2[1:3], axis=0), color="red", label="KDDCup", lw=3)
# plt.plot(data3[0], np.mean(data3[1:3], axis=0), color="orange", label="Amazon", lw=3)

N = 6
width = 0.25
fig, ax = plt.subplots(figsize=(8, 4))

ticklabels = ['1', '5', '10', '20', '50', '100']

p1 = ax.bar(np.arange(6), np.mean(data1[1:3], axis=0), width, hatch='/')
p2 = ax.bar(np.arange(6) + width, np.mean(data2[1:3], axis=0), width)
p3 = ax.bar(np.arange(6) + 2 * width, np.mean(data3[1:3], axis=0), width, hatch='.')
ax.set_xticks(np.arange(6) + width)
ax.set_xticklabels(ticklabels, fontsize=16)

ax.set_yticklabels(np.arange(0, 7, 1))
plt.setp(ax.get_yticklabels(), fontsize=16)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.ylim(0, 0.06)

totals = []

# find the values and append to list
for i in ax.patches:
    totals.append(i.get_height())

# set individual bar lables using above list
total = sum(totals)

# set individual bar lables using above list
for i in ax.patches:
    # get_x pulls left or right; get_height pushes up or down
    height = str((i.get_height() * 100))[0:4]
    if i.get_height() < 0.0009:
        height = "  0"
        ax.text(i.get_x() - 0.05, i.get_height(), height, fontsize=14, color='black')
    else:
        ax.text(i.get_x() - 0.05, i.get_height() + .001, height, fontsize=14, color='black')

# ##############################

plt.xlabel('Batch Size', fontsize=16)
plt.ylabel('Attack Rate (%)', fontsize=16)

ax.legend((p1[0], p2[0], p3[0]),
          ('MNIST', 'KDDCup', 'Amazon'),
          loc='best', ncol=3, fontsize=16)

fig.tight_layout(pad=0.1)
fig.savefig("fig_batch_bar.pdf")

plt.show()
