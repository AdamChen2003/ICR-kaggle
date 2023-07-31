from sklearn.svm import SVC
from process_data import getBinaryClassData, splitTrainAndTest, getMultiClassData
from evaluate_model import EvaluateModel

X,y = getBinaryClassData()
# X = pca(X)
X_train, y_train, X_test, y_test = splitTrainAndTest(X, y)
model = SVC()

grid = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    'kernel': ['rbf','linear','sigmoid','poly'],
    'degree': [2,3,4,5],
    'class_weight': ['balanced', None]
}

grid_os = {
    'classification__C': [0.1, 1, 10, 100, 1000],
    'classification__gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    'classification__kernel': ['rbf','linear','sigmoid','poly'],
    'classification__degree': [2,3,4,5]
}

# No sampling
EvaluateModel(X_train, y_train, X_test, y_test, model, grid, False)

# Oversampling
EvaluateModel(X_train, y_train, X_test, y_test, model, grid_os, True)

X,y = getMultiClassData()
X_train, y_train, X_test, y_test = splitTrainAndTest(X, y)

# No sampling
EvaluateModel(X_train, y_train, X_test, y_test, model, grid, False, multi=True)

# Oversampling
EvaluateModel(X_train, y_train, X_test, y_test, model, grid_os, True, multi=True)