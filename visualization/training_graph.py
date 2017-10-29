import pandas as pd
from pylab import *

training = pd.DataFrame.from_csv("training_history.tsv", sep='\t', index_col=None)
epochs = list(range(20))

fig2015, ax2015 = plt.subplots()
fig2016, ax2016 = plt.subplots()
fig2017, ax2017 = plt.subplots()
fig_all, ax_all = plt.subplots()
labels2015 = labels2016 = labels2017 = labels_all = []

for key, grp in training.groupby(['time']):
    if "2015" in key:
        ax2015 = grp.plot(ax=ax2015, kind='line', x='epoch', y='val_acc', label=key, linewidth=1)
        labels2015.append(key)
        ax2015.set_xticks(epochs)
        ax2015.set_xticklabels(epochs)
        ax2015.grid(True)
    elif "2016" in key:
        ax2016 = grp.plot(ax=ax2016, kind='line', x='epoch', y='val_acc', label=key, linewidth=1)
        labels2016.append(key)
        ax2016.set_xticks(epochs)
        ax2016.set_xticklabels(epochs)
        ax2016.grid(True)
    elif "2017" in key:
        ax2017 = grp.plot(ax=ax2017, kind='line', x='epoch', y='val_acc', label=key, linewidth=1)
        labels2017.append(key)
        ax2017.set_xticks(epochs)
        ax2017.set_xticklabels(epochs)
        ax2017.grid(True)
    else:
        ax_all = grp.plot(ax=ax_all, kind='line', x='epoch', y='val_acc', label=key, linewidth=1)
        labels_all.append(key)
        ax_all.set_xticks(epochs)
        ax_all.set_xticklabels(epochs)
        ax_all.grid(True)


lines2015, _ = ax2015.get_legend_handles_labels()
lines2016, _ = ax2016.get_legend_handles_labels()
lines2017, _ = ax2017.get_legend_handles_labels()
lines_all, _ = ax_all.get_legend_handles_labels()

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()
