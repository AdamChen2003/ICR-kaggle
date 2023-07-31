import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from process_data import getBinaryClassData, Normalize, pca
from evaluate_model import EvaluateModel
from imblearn.over_sampling import SMOTE

X,y = getBinaryClassData()
# X = pca(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify=y, shuffle=True, random_state=1)
X_train, X_test = Normalize(X_train, X_test)
X_train = np.nan_to_num(X_train)
X_test = np.nan_to_num(X_test)
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