from sklearn.linear_model import LogisticRegression
from process_data import getBinaryClassData, splitTrainAndTest, getMultiClassData
from evaluate_model import EvaluateModel

X,y = getBinaryClassData()
# X = pca(X)
X_train, y_train, X_test, y_test = splitTrainAndTest(X, y)
model = LogisticRegression(random_state=1)


grid = {
    'C': [1,0.1,0.01],
    'fit_intercept': [True,False],
    'class_weight': ['balanced', None]
}

grid_os = {
    'classification__C': [1,0.1,0.01],
    'classification__fit_intercept': [True,False]
}


# No sampling/Class Weights
EvaluateModel(X_train, y_train, X_test, y_test, model, grid, False)

# Oversampling
EvaluateModel(X_train, y_train, X_test, y_test, model, grid_os, True)

X,y = getMultiClassData()
X_train, y_train, X_test, y_test = splitTrainAndTest(X, y)

# Multi class with no sampling
EvaluateModel(X_train, y_train, X_test, y_test, model, grid, False, multi=True)

# Multi class with oversampling
EvaluateModel(X_train, y_train, X_test, y_test, model, grid_os, True, multi=True)