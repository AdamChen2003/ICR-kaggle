from statistics import stdev
import numpy as np
from sklearn.model_selection import StratifiedKFold
from process_data import Normalize

def EvaluateModel(X, y, model):
    skf = StratifiedKFold(n_splits=10,shuffle=True)
    scores = []
    for train_index, test_index in skf.split(X,y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_train, X_test = Normalize(X_train, X_test)
        X_train = np.nan_to_num(X_train)
        X_test = np.nan_to_num(X_test)
        model.fit(X_train, y_train.ravel())
        scores.append(model.score(X_test, y_test.ravel()))

    print(f'Maximum Accuracy Score: {max(scores)}')
    print(f'Minimum Accuracy Score: {min(scores)}')
    print(f'Mean Accuracy Score: {np.mean(scores)}')
    print(f'Standard Deviation: {stdev(scores)}')