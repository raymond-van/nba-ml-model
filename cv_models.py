from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from data import X, y
from train import rf, xgb, gb, et, sv

kfold = StratifiedKFold(n_splits=3, random_state=17)

rf_results = cross_val_score(rf, X, y, cv=kfold, n_jobs=4)
xgb_results = cross_val_score(xgb, X, y, cv=kfold, n_jobs=4)
gb_results = cross_val_score(gb, X, y, cv=kfold, n_jobs=4)
et_results = cross_val_score(et, X, y, cv=kfold, n_jobs=4)
sv_results = cross_val_score(sv, X, y, cv=kfold, n_jobs=4)

print("Accuracy: %.2f%% (%.2f%%)" % (rf_results.mean()*100, rf_results.std()*100))
print("Accuracy: %.2f%% (%.2f%%)" % (xgb_results.mean()*100, xgb_results.std()*100))
print("Accuracy: %.2f%% (%.2f%%)" % (gb_results.mean()*100, gb_results.std()*100))
print("Accuracy: %.2f%% (%.2f%%)" % (et_results.mean()*100, et_results.std()*100))
print("Accuracy: %.2f%% (%.2f%%)" % (sv_results.mean()*100, sv_results.std()*100))

# RF Accuracy: 94.52% (2.09%)
# XGB Accuracy: 96.35% (1.72%)
# GB Accuracy: 97.26% (1.00%)
# ET Accuracy: 90.12% (1.13%)
# SV Accuracy: 95.89% (1.13%)

# after hyperparameter tuning:
# RF Accuracy: 93.92% (2.06%)
# XGB Accuracy: 96.35% (1.72%)
# GB Accuracy: 96.96% (0.95%)
# ET Accuracy: 93.77% (1.13%)
# SV Accuracy: 98.93% (0.57%)

# after hyperparameter tuning rerun:
# RF Accuracy: 94.92% (0.17%)
# XGB Accuracy: 96.81% (1.16%)
# GB Accuracy: 96.57% (1.02%)
# ET Accuracy: 94.10% (0.45%)
# SV Accuracy: 98.11% (0.60%)

# Ensemble Accuracy: 97.83% (0.80%)