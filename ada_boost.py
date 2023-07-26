from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from process_data import getBinaryClassData, getBinaryClassWeights, getMultiClassData, getMultiClassWeights

X, y = getBinaryClassData()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify=y, shuffle=True)
X_train, X_test = Normalize(X_train, X_test)
X_train = np.nan_to_num(X_train)
X_test = np.nan_to_num(X_test)

# model = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1, criterion='entropy',class_weight=getBinaryClassWeights()),n_estimators=1000)
# EvaluateModel(X, y, model)

# model = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1, criterion='entropy',class_weight=getMultiClassWeights()),n_estimators=1000)
# EvaluateModel(X, y, model, True)

from statistics import stdev
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score
from process_data import Normalize

# multiclass = True
# X,y = getMultiClassData()
# skf = StratifiedKFold(n_splits=5,shuffle=True)
# scores = []
# f1_scores = []
# for train_index, test_index in skf.split(X,y):
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#     X_train, X_test = Normalize(X_train, X_test)
#     X_train = np.nan_to_num(X_train)
#     X_test = np.nan_to_num(X_test)
#     model = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1, criterion='entropy',class_weight=getMultiClassWeights(y_train)),n_estimators=1000)
#     model.fit(X_train, y_train.ravel())
#     scores.append(model.score(X_test, y_test.ravel()))
#     if multiclass:
#         pred = model.predict(X_test)
#         f1_scores.append(f1_score(np.where(y_test > 0, 1, y_test), np.where(pred > 0, 1, pred), average='micro'))
#     else:
#         f1_scores.append(f1_score(y_test, model.predict(X_test), average='micro'))

# print(f'Maximum Accuracy Score: {max(scores)}')
# print(f'Minimum Accuracy Score: {min(scores)}')
# print(f'Mean Accuracy Score: {np.mean(scores)}')
# print(f'Accuracy Standard Deviation: {stdev(scores)}')
# print()
# print(f'Maximum F1 Score: {max(f1_scores)}')
# print(f'Minimum F1 Score: {min(f1_scores)}')
# print(f'Mean F1 Score: {np.mean(f1_scores)}')
# print(f'F1 Standard Deviation: {stdev(f1_scores)}')

# base_models = []

# for i in [1,3,6,8]:
#     base_models.append(DecisionTreeClassifier(max_depth=1, criterion='entropy',class_weight=getMultiClassWeights(y_train)))

# grid = {
#     'estimator': [],
#     'criterion': ['gini','entropy'],
#     'n_estimators': [200,500,1000]
# }