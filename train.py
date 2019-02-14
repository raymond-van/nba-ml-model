import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC 
from xgboost import XGBClassifier
from data_unprocessed import X, y

# Train models on full dataset to use for predictions
rf = RandomForestClassifier(n_estimators=1000,min_samples_leaf=2,min_samples_split=5,max_depth=20)
rf.fit(X,y)

xgb = XGBClassifier(n_estimators=130,min_child_weight=2,max_depth=5,colsample_bytree=.7,learning_rate=.125)
xgb.fit(X,y)

gb = GradientBoostingClassifier(n_estimators=600,min_samples_leaf=5,max_depth=80)
gb.fit(X,y)

et = ExtraTreesClassifier(n_estimators=1400,min_samples_split=5,min_samples_leaf=1,max_depth=30)
et.fit(X,y)

sv = SVC(kernel='poly',gamma=3,degree=1,C=1,probability=True)
sv.fit(X,y)

# Validation accuracy 

# x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=17)

# rf = RandomForestClassifier(n_estimators=1000,min_samples_leaf=1,min_samples_split=5,max_depth=80)
# rf.fit(x_train,y_train)
# rf_pred = rf.predict(x_val)
# rf_acc = np.mean(rf_pred==y_val)

# xgb = XGBClassifier(n_estimators=150,min_child_weight=1,max_depth=9,colsample_bytree=.5,learning_rate=.15)
# xgb.fit(x_train,y_train)
# xgb_pred = xgb.predict(x_val)
# xgb_acc = np.mean(xgb_pred==y_val)

# gb = GradientBoostingClassifier()
# gb.fit(x_train,y_train)
# gb_pred = gb.predict(x_val)
# gb_acc = np.mean(gb_pred==y_val)

# et = ExtraTreesClassifier()
# et.fit(x_train,y_train)
# et_pred = et.predict(x_val)
# et_acc = np.mean(et_pred==y_val)

# sv = SVC()
# sv.fit(x_train,y_train)
# sv_pred = sv.predict(x_val)
# sv_acc = np.mean(sv_pred==y_val)

# print(rf_acc,xgb_acc,gb_acc,et_acc,sv_acc)