from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from process_data import getBinaryClassData, getMultiClassData
from evaluate_model import EvaluateModel

X,y = getBinaryClassData()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify=y, shuffle=True, random_state=1)
model = KNeighborsClassifier()

grid = {
    'classification__n_neighbors': [2,4,6,8,10],
    'classification__weights': ['uniform','distance']
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify=y, shuffle=True, random_state=1)

# Multi class with no sampling
EvaluateModel(X_train, y_train, X_test, y_test, model, grid, False, multi=True)

# Multi class with oversampling
EvaluateModel(X_train, y_train, X_test, y_test, model, grid_os, True, multi=True)