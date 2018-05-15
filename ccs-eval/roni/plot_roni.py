import numpy as np 
import matplotlib.pyplot as plt
import pdb

fig, ax = plt.subplots(figsize=(10, 5))

data = np.load("all_roni_scores_0.npy")

toplot = np.sum(data, axis=0) * -1

colors = []

for i in range(10):
    colors.append("blue")

for i in range(5):
    colors.append("red")

plt.bar(np.arange(15), toplot, color=colors)
plt.xlabel("Client Label", fontsize=18)
plt.ylabel("Final RONI Score", fontsize=18)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.setp(ax.get_xticklabels(), fontsize=18)
plt.setp(ax.get_yticklabels(), fontsize=18)
plt.axvline(x=9.5, color='k', linestyle='--')

fig.tight_layout(pad=0.1)
plt.savefig("fig_roni.pdf")
plt.show()

