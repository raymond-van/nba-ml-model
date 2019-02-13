import numpy as np
import matplotlib.pyplot as plt
from data import X, y, mean_team_stats
from train import rf, xgb

importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print features and associated index number
for i,feat in enumerate(X.columns):
     print(i, feat)

# Print the feature ranking
print("Feature ranking:")
for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()


# Unimportant columns to drop: 'AST_PCTA','AST_PCTB','BLKA','BLKB','FTA_RATEA','FTA_RATEB','FT_PCTA','FT_PCTB','OPP_FTA_RATEA','OPP_FTA_RATEB','PACEA','PACEB','PFA','PFB','STLA','STLB','OREBA','OREBB'