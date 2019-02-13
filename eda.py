import pandas as pd
import numpy as np
from data import X,y,mean_team_stats
import os
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt

# Win percentage with HCA
raw_data = pd.read_feather(os.getcwd() + '/data/nba_data.feather')
print(raw_data.groupby(['TEAMA_HCA'])['TEAMA_WIN'].agg([np.mean])) # HCA increases win % by a significant margin

# Correlation heatmap
corr_matrix = X.corr()
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(corr_matrix,xticklabels=True, yticklabels=True,ax=ax)
plt.show()