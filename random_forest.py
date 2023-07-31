from sklearn.ensemble import RandomForestClassifier
from process_data import getBinaryClassData, splitTrainAndTest, getMultiClassData
from evaluate_model import EvaluateModel

X,y = getBinaryClassData()
# X = pca(X)
X_train, y_train, X_test, y_test = splitTrainAndTest(X, y)
model = RandomForestClassifier()

grid = {
    'n_estimators': [200,400,800,1000],
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': [1,3,5,7],
    'class_weight': ['balanced', None]
}

grid_os = {
    'n_estimators': [200,400,800,1000],
    'classification__criterion': ['gini', 'entropy', 'log_loss'],
    'classification__max_depth': [1,3,5,7]
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