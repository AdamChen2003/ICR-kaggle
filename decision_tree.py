import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from process_data import getBinaryClassData, Normalize
from evaluate_model import EvaluateModel
from imblearn.over_sampling import SMOTE

X,y = getBinaryClassData()
# X = pca(X)
X_train, y_train, X_test, y_test = splitTrainAndTest(X, y)
model = DecisionTreeClassifier(random_state=1)

# No sampling
grid = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': [2, 3, 5, 10, 20],
    'min_samples_leaf': [5, 10, 20, 50, 100],
    'class_weight': ['balanced', None]
}

EvaluateModel(X_train, y_train, X_test, y_test, model, grid, False)

# Oversampling
grid = {
    'classification__criterion': ['gini', 'entropy', 'log_loss'],
    'classification__max_depth': [2, 3, 5, 10, 20],
    'classification__min_samples_leaf': [5, 10, 20, 50, 100]
}

EvaluateModel(X_train, y_train, X_test, y_test, model, grid, True)