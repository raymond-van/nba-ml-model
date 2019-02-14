import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC 
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, precision_score, recall_score, f1_score
from data_unprocessed import X,y
import warnings
warnings.filterwarnings("ignore")

# Base models
rf = RandomForestClassifier(n_estimators=1000,min_samples_leaf=2,min_samples_split=5,max_depth=20)
xgb = XGBClassifier(n_estimators=130,min_child_weight=2,max_depth=5,colsample_bytree=.7,learning_rate=.125)
gb = GradientBoostingClassifier(n_estimators=600,min_samples_leaf=5,max_depth=80)
et = ExtraTreesClassifier(n_estimators=1400,min_samples_split=5,min_samples_leaf=1,max_depth=30)
sv = SVC(kernel='poly',gamma=3,degree=1,C=1,probability=True)

# Calibrated models used for predictions
rf_isotonic = CalibratedClassifierCV(rf, cv=3, method='isotonic')
rf_isotonic.fit(X,y)
xgb_sigmoid = CalibratedClassifierCV(xgb, cv=3, method='sigmoid')
xgb_sigmoid.fit(X,y)
gb_sigmoid = CalibratedClassifierCV(gb, cv=3, method='sigmoid')
gb_sigmoid.fit(X,y)
sv_sigmoid = CalibratedClassifierCV(sv, cv=3, method='sigmoid')
sv_sigmoid.fit(X,y)
et_isotonic = CalibratedClassifierCV(et, cv=3, method='isotonic')
et_isotonic.fit(X,y)

# Create reliability plots

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=19)

# def plot_calibration_curve(est, name, fig_index):
#     """Plot calibration curve for est w/o and with calibration. """
#     # Calibrated with isotonic calibration
#     isotonic = CalibratedClassifierCV(est, cv=3, method='isotonic')

#     # Calibrated with sigmoid calibration
#     sigmoid = CalibratedClassifierCV(est, cv=3, method='sigmoid')

#     fig = plt.figure(fig_index, figsize=(10, 10))
#     ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
#     ax2 = plt.subplot2grid((3, 1), (2, 0))

#     ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
#     for clf, name in [(est, name),
#                       (isotonic, name + ' + Isotonic'),
#                       (sigmoid, name + ' + Sigmoid')]:
#         clf.fit(X_train, y_train)
#         y_pred = clf.predict(X_test)
#         if hasattr(clf, "predict_proba"):
#             prob_pos = clf.predict_proba(X_test)[:, 1]
#         else:  # use decision function
#             prob_pos = clf.decision_function(X_test)
#             prob_pos = \
#                 (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

#         clf_score = brier_score_loss(y_test, prob_pos, pos_label=y.max())
#         print("%s:" % name)
#         print("\tBrier: %1.3f" % (clf_score))
#         print("\tPrecision: %1.3f" % precision_score(y_test, y_pred))
#         print("\tRecall: %1.3f" % recall_score(y_test, y_pred))
#         print("\tF1: %1.3f\n" % f1_score(y_test, y_pred))

#         fraction_of_positives, mean_predicted_value = \
#             calibration_curve(y_test, prob_pos, n_bins=5)

#         ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
#                  label="%s (%1.3f)" % (name, clf_score))

#         ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
#                  histtype="step", lw=2)

#     ax1.set_ylabel("Fraction of positives")
#     ax1.set_ylim([-0.05, 1.05])
#     ax1.legend(loc="lower right")
#     ax1.set_title('Calibration plots  (reliability curve)')

#     ax2.set_xlabel("Mean predicted value")
#     ax2.set_ylabel("Count")
#     ax2.legend(loc="upper center", ncol=2)

#     plt.tight_layout()

# plot_calibration_curve(rf, "Random Forest", 1)
# plot_calibration_curve(xgb, "XGB", 2)
# plot_calibration_curve(gb, "Gradient Boosted", 3)
# plot_calibration_curve(sv, "SVC", 4)
# plot_calibration_curve(et, "Extra Trees", 5)

# plt.show()