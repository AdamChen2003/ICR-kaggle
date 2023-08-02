from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from process_data import getBinaryClassData, getMultiClassData
from evaluate_model import EvaluateModel

X,y = getBinaryClassData()
# X = pca(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify=y, shuffle=True, random_state=1)
model = LogisticRegression(random_state=1)

grid = {
    'classification__C': [1,0.1,0.01],
    'classification__fit_intercept': [True,False],
    'classification__class_weight': ['balanced', None]
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify=y, shuffle=True, random_state=1)

# # Multi class with no sampling
# EvaluateModel(X_train, y_train, X_test, y_test, model, grid, False, multi=True)

# # Multi class with oversampling
# EvaluateModel(X_train, y_train, X_test, y_test, model, grid_os, True, multi=True)