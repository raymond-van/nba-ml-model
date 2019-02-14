import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC 
from data_unprocessed import X, y
from ensemble import xgb_stack
from train import rf, xgb, gb, et, sv
from predict import today_df

# Base learners which consists of these classifiers: Random Forests, Gradient Boosted, Extra Trees, Support Vector Machines
x_train = X.values
x_test = today_df.values
y_train = y.values
ntrain = x_train.shape[0]
ntest = x_test.shape[0]
SEED = 17
NFOLDS = 3
kf = KFold(n_splits=NFOLDS, random_state=SEED)

def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))
    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]
        clf.fit(x_tr, y_tr)
        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)
    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

rf_oof_train, rf_oof_test = get_oof(rf, x_train, y_train, x_test)
et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test)
gb_oof_train, gb_oof_test = get_oof(gb, x_train, y_train, x_test)
sv_oof_train, sv_oof_test = get_oof(sv, x_train, y_train, x_test)

# Data for meta learner
x_train = np.concatenate((et_oof_train, rf_oof_train, gb_oof_train, sv_oof_train), axis=1)
x_test = np.concatenate((et_oof_test, rf_oof_test, gb_oof_test, sv_oof_test), axis=1)

# Train and then predict with XGB as meta learner
xgb_stack = RandomForestClassifier(n_estimators=1000,min_samples_leaf=1,min_samples_split=5,max_depth=80)
xgb_stack.fit(x_train,y_train)
preds = xgb_stack.predict_proba(x_test)
print(preds)