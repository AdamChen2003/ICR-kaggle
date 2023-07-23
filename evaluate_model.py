from statistics import stdev
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from process_data import Normalize

def EvaluateModel(X, y, model, multiclass=False):
    skf = StratifiedKFold(n_splits=5,shuffle=True)
    scores = []
    f1_scores = []
    for train_index, test_index in skf.split(X,y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_train, X_test = Normalize(X_train, X_test)
        X_train = np.nan_to_num(X_train)
        X_test = np.nan_to_num(X_test)
        model.fit(X_train, y_train.ravel())
        scores.append(model.score(X_test, y_test.ravel()))
        if multiclass:
            pred = model.predict(X_test)
            f1_scores.append(f1_score(np.where(y_test > 0, 1, y_test), np.where(pred > 0, 1, pred), average='micro'))
        else:
            f1_scores.append(f1_score(y_test, model.predict(X_test), average='micro'))

    print(f'Maximum Accuracy Score: {max(scores)}')
    print(f'Minimum Accuracy Score: {min(scores)}')
    print(f'Mean Accuracy Score: {np.mean(scores)}')
    print(f'Accuracy Standard Deviation: {stdev(scores)}')
    print()
    print(f'Maximum F1 Score: {max(f1_scores)}')
    print(f'Minimum F1 Score: {min(f1_scores)}')
    print(f'Mean F1 Score: {np.mean(f1_scores)}')
    print(f'F1 Standard Deviation: {stdev(f1_scores)}')