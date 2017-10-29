import pandas as pd
import seaborn as sn
from pylab import *

cm = pd.DataFrame.from_csv("../tf_models/tf_rnn_embeddings/tf_prediction_results/cf_matrix.tsv", sep='\t', index_col=None)


sections = ["2015_1q", "2015_2q", "2015_3q", "2015_4q", "2016_1q", "2016_2q", "2016_3q", "2016_4q", "2017_1q", "2017_2q"]

matrix = []
for key, grp in cm.groupby(['model_date']):
    matrix.append(list(grp["accuracy"]))

df_cm = pd.DataFrame(matrix, index = [i for i in sections],columns = [i for i in sections])

plt.figure(figsize = (10,7))
plt.xticks(rotation=30)
yticks = df_cm.index
sn.heatmap(df_cm, yticklabels=yticks, annot=True)
plt.yticks(rotation=0)
plt.xlabel("Data by quarter")
plt.ylabel("Models by quarter")
plt.title("Confusion matrix with prediction accuracy accross all data and models")
plt.show()

