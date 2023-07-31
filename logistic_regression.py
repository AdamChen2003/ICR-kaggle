import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from process_data import getBinaryClassData, Normalize, Standardize, pca, splitTrainAndTest
from evaluate_model import EvaluateModel
from imblearn.over_sampling import SMOTE

X,y = getBinaryClassData()
# X = pca(X)
X_train, y_train, X_test, y_test = splitTrainAndTest(X, y)
model = LogisticRegression(random_state=1)

# No sampling/Class Weights
grid = {
    'C': [1,0.1,0.01],
    'fit_intercept': [True,False],
    'class_weight': ['balanced', None]
}

EvaluateModel(X_train, y_train, X_test, y_test, model, grid, False)

# Oversampling
grid = {
    'classification__C': [1,0.1,0.01],
    'classification__fit_intercept': [True,False]
}

EvaluateModel(X_train, y_train, X_test, y_test, model, grid, True)