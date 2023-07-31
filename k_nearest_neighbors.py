from sklearn.neighbors import KNeighborsClassifier
from process_data import getBinaryClassData, splitTrainAndTest, getMultiClassData
from evaluate_model import EvaluateModel

X,y = getBinaryClassData()
# X = pca(X)
X_train, y_train, X_test, y_test = splitTrainAndTest(X, y)
model = KNeighborsClassifier()

grid = {
    'n_neighbors': [2,4,6,8,10],
    'weights': ['uniform','distance']
}

grid_os = {
    'classification__n_neighbors': [2,4,6,8,10],
    'classification__weights': ['uniform','distance']
}

# No sampling
EvaluateModel(X_train, y_train, X_test, y_test, model, grid, False)

# Oversampling
EvaluateModel(X_train, y_train, X_test, y_test, model, grid_os, True)

X,y = getMultiClassData()
X_train, y_train, X_test, y_test = splitTrainAndTest(X, y)

# Multi class with no sampling
EvaluateModel(X_train, y_train, X_test, y_test, model, grid, False, multi=True)

# Multi class with oversampling
EvaluateModel(X_train, y_train, X_test, y_test, model, grid_os, True, multi=True)