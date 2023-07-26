import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from process_data import getBinaryClassData, getBinaryClassWeights, Normalize
from imblearn.over_sampling import SMOTE

X,y = getBinaryClassData()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify=y, shuffle=True)
X_train, X_test = Normalize(X_train, X_test)
X_train = np.nan_to_num(X_train)
X_test = np.nan_to_num(X_test)

# Class Weights
grid = {
    'C': [0.5,1.0,1.5],
    'fit_intercept': [True,False],
    'class_weight': [getBinaryClassWeights(y_train), None]
}

cv = GridSearchCV(estimator=LogisticRegression(), param_grid=grid, cv= 5)
cv.fit(X_train, y_train.ravel())
print(cv.best_params_)
print(cv.best_score_)
print(cv.best_estimator_.score(X_test, y_test.ravel()))


# Oversampling
oversample = SMOTE()
X_train, y_train = oversample.fit_resample(X_train, y_train)

grid = {
    'C': [0.5,1.0,1.5],
    'fit_intercept': [True,False]
}

cv = GridSearchCV(estimator=LogisticRegression(), param_grid=grid, cv= 5)
cv.fit(X_train, y_train.ravel())
print(cv.best_params_)
print(cv.best_score_)
print(cv.best_estimator_.score(X_test, y_test.ravel()))