from statistics import stdev
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from process_data import getBinaryClassData, Normalize, getBinaryClassWeights

X,y = getBinaryClassData()

skf = StratifiedKFold(n_splits=5,shuffle=True)
model = AdaBoostClassifier(
                estimator=
                DecisionTreeClassifier(max_depth=1, criterion='entropy',class_weight=getBinaryClassWeights()),
                n_estimators=1000)

scores = []
for train_index, test_index in skf.split(X,y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    X_train = np.nan_to_num(Normalize(X_train, X_train))
    X_test = np.nan_to_num(Normalize(X_train, X_test))
    model.fit(X_train, y_train.ravel())
    scores.append(model.score(X_test, y_test.ravel()))

print(f'Maximum Accuracy Score: {max(scores)}')
print(f'Minimum Accuracy Score: {min(scores)}')
print(f'Mean Accuracy Score: {np.mean(scores)}')
print(f'Standard Deviation: {stdev(scores)}')
