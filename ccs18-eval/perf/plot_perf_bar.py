import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pdb

df = pd.read_csv("perf.csv", header=None)
data = df.values

N = 5
width = 0.25
fig, ax = plt.subplots(figsize=(8, 4))

ticklabels = ['10', '20', '30', '40', '50']

p1 = ax.bar(np.arange(5), data[2,:] / data[1,:], width, hatch='/')
p2 = ax.bar(np.arange(5) + width, data[4,:] / data[3,:], width, hatch='+')
ax.set_xticks(np.arange(5) + (width / 2))
ax.set_xticklabels(ticklabels, fontsize=16)
ax.set_yticklabels(['0', '0.25', '0.5', '0.75', '1', '1.25', '1.5', '1.75'],
	fontsize=16)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.ylim(0, 2)

totals = []

# # find the values and append to list
# for i in ax.patches:
#     totals.append(i.get_height())

# # set individual bar lables using above list
# total = sum(totals)

# # set individual bar lables using above list
# for i in ax.patches:
#     # get_x pulls left or right; get_height pushes up or down
#     height = str((i.get_height() * 100))[0:4]
#     if i.get_height() < 0.0009:
#         height = "  0"
#         ax.text(i.get_x() - 0.05, i.get_height(), height, fontsize=14, color='black')
#     else:
#         ax.text(i.get_x() - 0.05, i.get_height() + .001, height, fontsize=14, color='black')

# # ##############################

plt.xlabel('# of Clients', fontsize=16)
plt.ylabel('Relative Slowdown', fontsize=16)

ax.legend((p1[0], p2[0]),
          ('MNIST (CPU)', 'VGGFace2 (GPU)'),
          loc='best', ncol=1, fontsize=16)

fig.tight_layout(pad=0.1)
fig.savefig("fig_perf_relative.pdf")

plt.show()
