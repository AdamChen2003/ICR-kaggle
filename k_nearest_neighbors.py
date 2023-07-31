import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from process_data import getBinaryClassData, Normalize, pca
from evaluate_model import EvaluateModel
from imblearn.over_sampling import SMOTE

X,y = getBinaryClassData()
# X = pca(X)
X_train, y_train, X_test, y_test = splitTrainAndTest(X, y)
model = KNeighborsClassifier()

# No sampling
grid = {
    'n_neighbors': [2,4,6,8,10],
    'weights': ['uniform','distance']
}

EvaluateModel(X_train, y_train, X_test, y_test, model, grid, False)

# Oversampling

grid = {
    'classification__n_neighbors': [2,4,6,8,10],
    'classification__weights': ['uniform','distance']
}

EvaluateModel(X_train, y_train, X_test, y_test, model, grid, True)