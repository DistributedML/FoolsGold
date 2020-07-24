import numpy as np

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt
import pandas as pd
import pdb


df1 = pd.read_csv("combined_attack_fg.csv", header=None)
data1 = df1.values

df2 = pd.read_csv("combined_attack_kr.csv", header=None)
data2 = df2.values

df3 = pd.read_csv("combined_attack_krfg.csv", header=None)
data3 = df3.values

width = 0.25
fig, ax = plt.subplots(figsize=(8, 2))
ticklabels = ['FoolsGold', 'MultiKrum', 'FoolsGold+MultiKrum']

plot_data1 = np.ones(3)
plot_data1[0] = np.mean(data1[:,0])
plot_data1[1] = np.mean(data2[:,0])
plot_data1[2] = np.mean(data3[:,0])

plot_data2 = np.ones(3)
plot_data2[0] = np.mean(data1[:,4])
plot_data2[1] = np.mean(data2[:,4])
plot_data2[2] = np.mean(data3[:,4])

p1 = ax.bar(np.arange(3), plot_data1, width, hatch='/')
p2 = ax.bar(np.arange(3) + width, plot_data2, width, hatch='.')

ax.set_xticks(np.arange(3) + width / 2)
ax.set_xticklabels(ticklabels, fontsize=14)

ax.set_yticklabels(np.array([0, 50, 100]))
plt.setp(ax.get_yticklabels(), fontsize=14)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.ylim(0, 1.8)

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
        ax.text(i.get_x(), i.get_height(), height, fontsize=14, color='black')
    else:
        ax.text(i.get_x() + 0.05, i.get_height() + .02, height, fontsize=14, color='black')

# ##############################
plt.ylabel('%', fontsize=16)

ax.legend((p1[0], p2[0]),
          ('Test Accuracy', 'Attack Rate'),
          #loc='best',
          #bbox_to_anchor=(0.8, 1),
          ncol=2, fontsize=16)

fig.tight_layout(pad=0.1)
fig.savefig("fig_krum_combined.pdf")

plt.show()
