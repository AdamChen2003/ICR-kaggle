from sklearn.tree import DecisionTreeClassifier
from process_data import getBinaryClassData, splitTrainAndTest, getMultiClassData
from evaluate_model import EvaluateModel


X,y = getBinaryClassData()
# X = pca(X)
X_train, y_train, X_test, y_test = splitTrainAndTest(X, y)
model = DecisionTreeClassifier(random_state=1)

grid = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': [2, 3, 5, 10, 20],
    'min_samples_leaf': [5, 10, 20, 50, 100],
    'class_weight': ['balanced', None]
}

grid_os = {
    'classification__criterion': ['gini', 'entropy', 'log_loss'],
    'classification__max_depth': [2, 3, 5, 10, 20],
    'classification__min_samples_leaf': [5, 10, 20, 50, 100]
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