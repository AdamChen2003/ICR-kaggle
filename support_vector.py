import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from process_data import getBinaryClassData, Normalize, pca
from evaluate_model import EvaluateModel
from imblearn.over_sampling import SMOTE

X,y = getBinaryClassData()
# X = pca(X)
X_train, y_train, X_test, y_test = splitTrainAndTest(X, y)
model = SVC()

# No sampling
grid = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    'kernel': ['rbf','linear','sigmoid','poly'],
    'degree': [2,3,4,5],
    'class_weight': ['balanced', None]
}

EvaluateModel(X_train, y_train, X_test, y_test, model, grid, False)

# Oversampling
grid = {
    'classification__C': [0.1, 1, 10, 100, 1000],
    'classification__gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    'classification__kernel': ['rbf','linear','sigmoid','poly'],
    'classification__degree': [2,3,4,5]
}

EvaluateModel(X_train, y_train, X_test, y_test, model, grid, True)