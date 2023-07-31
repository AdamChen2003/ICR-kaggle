import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from process_data import getBinaryClassData, Normalize, pca
from evaluate_model import EvaluateModel
from imblearn.over_sampling import SMOTE

X,y = getBinaryClassData()
# X = pca(X)
X_train, y_train, X_test, y_test = splitTrainAndTest(X, y)
model = RandomForestClassifier()

# No sampling
grid = {
    'n_estimators': [200,400,800,1000],
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': [1,3,5,7],
    'class_weight': ['balanced', None]
}

EvaluateModel(X_train, y_train, X_test, y_test, model, grid, False)

# Oversampling
grid = {
    'n_estimators': [200,400,800,1000],
    'classification__criterion': ['gini', 'entropy', 'log_loss'],
    'classification__max_depth': [1,3,5,7]
}

EvaluateModel(X_train, y_train, X_test, y_test, model, grid, True)