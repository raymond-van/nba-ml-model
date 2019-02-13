import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC 
from sklearn.model_selection import RandomizedSearchCV
from data import X, y

min_samples_leaf = [1, 2, 3, 4, 5]
max_depth = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None]
n_estimators = [600,700,800,900,1000]

gb_cv_grid = {'n_estimators': n_estimators,
               'min_samples_leaf': min_samples_leaf,
               'max_depth': max_depth}
gb_cv = GradientBoostingClassifier()
gb_cv = RandomizedSearchCV(estimator=gb_cv, param_distributions=gb_cv_grid, n_iter=100,
                               cv=3, verbose=2, random_state=17, n_jobs=4)

gb_cv.fit(X,y)

################################################################################################

min_samples_split = [2,5,10]
min_samples_leaf = [1, 2, 3, 4, 5]
max_depth = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None]
n_estimators = [900,1000,1100,1200,1300,1400]
et_cv_grid = {'n_estimators': n_estimators,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'max_depth': max_depth}

et_cv = ExtraTreesClassifier()
et_cv = RandomizedSearchCV(estimator=et_cv, param_distributions=et_cv_grid, n_iter=200,
                               cv=3, verbose=2, random_state=17, n_jobs=4)

et_cv.fit(X,y)

################################################################################################

kernel = ['rbf', 'poly']
gamma = [0.1, 1, 3, 5, 10]
C = [0.1, 1, 3, 5, 10, 100]
degree = [1, 2, 3, 4, 5, 6]
sv_cv_grid = {'kernel': kernel,
               'gamma': gamma,
               'degree': degree,
               'C': C}

sv_cv = SVC()
sv_cv = RandomizedSearchCV(estimator=sv_cv, param_distributions=sv_cv_grid, n_iter=300,
                               cv=3, verbose=2, random_state=17, n_jobs=4)

sv_cv.fit(X,y)

################################################################################################

min_samples_leaf = [1, 2, 3, 4, 5]
max_depth = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None]
min_samples_split = [2, 5, 10]
n_estimators = [1000]
rf_cv_grid = {'n_estimators': n_estimators,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'max_depth': max_depth}

rf_cv = RandomForestClassifier()
rf_cv = RandomizedSearchCV(estimator=rf_cv, param_distributions=rf_cv_grid, n_iter=100,
                               cv=3, verbose=2, random_state=17, n_jobs=4)

rf_cv.fit(X,y)

################################################################################################

booster = ['gbtree','gblinear']
learning_rate = [0.05,0.075,0.1,0.125,0.15]
max_depth=[3,4,5,6,7,8,9]
n_estimators=[90,100,110,120,130,140,150]
min_child_weight=[1,2,3]
colsample_bytree=[.5,.6,.7,.8,.9,1]
xgb_cv_grid = {'learning_rate':learning_rate,
                   'max_depth': max_depth,
                   'n_estimators': n_estimators,
                   'min_child_weight': min_child_weight,
                   'colsample_bytree': colsample_bytree}

xgb_cv = XGBClassifier()
xgb_cv = RandomizedSearchCV(estimator=xgb_cv, param_distributions=xgb_cv_grid, n_iter=300,
                               cv=3, verbose=2, random_state=17, n_jobs=4)

xgb_cv.fit(X,y)

################################################################################################
 
print("GB PARAMS")
print(gb_cv.best_params_)
# Att 1: {'n_estimators': 900, 'min_samples_leaf': 4, 'max_depth': 30}
#{'n_estimators': 600, 'min_samples_leaf': 5, 'max_depth': 80}

print("ET PARAMS")
print(et_cv.best_params_)
# Attempt 1: {'n_estimators': 1000, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_depth': 30}
# Attempt 2: {'n_estimators': 1200, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_depth': 60}
# Attempt 3: {'n_estimators': 1100, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_depth': 50}
# {'n_estimators': 1400, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_depth': 30}

print("SV PARAMS")
print(sv_cv.best_params_)
# Attempt 1: {'kernel': 'poly', 'gamma': 3, 'degree': 1, 'C': 1}
# Attempt 2: {'kernel': 'poly', 'gamma': 3, 'degree': 1, 'C': 1}
# {'kernel': 'poly', 'gamma': 3, 'degree': 1, 'C': 1}

print("RF PARAMS")
print(rf_cv.best_params_)
# {'n_estimators': 1000, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_depth': 20}

print("XGB PARAMS")
print(xgb_cv.best_params_)
# {'n_estimators': 130, 'min_child_weight': 2, 'max_depth': 5, 'learning_rate': 0.125, 'colsample_bytree': 0.7}