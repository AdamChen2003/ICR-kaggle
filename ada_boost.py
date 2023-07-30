import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from process_data import getBinaryClassData, Normalize
from evaluate_model import EvaluateModel
from imblearn.over_sampling import SMOTE

X,y = getBinaryClassData()
# X = pca(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify=y, shuffle=True, random_state=1)
X_train, X_test = Normalize(X_train, X_test)
X_train = np.nan_to_num(X_train)
X_test = np.nan_to_num(X_test)
model = AdaBoostClassifier(random_state=1)

# No sampling
base_models = []

for i in [1,2,3]:
    base_models.append(DecisionTreeClassifier(max_depth=1, criterion='entropy'))

grid = {
    'estimator': base_models,
    'n_estimators': [200,500,1000]
}

EvaluateModel(X_train, y_train, X_test, y_test, model, grid)

# Oversampling
oversample = SMOTE()
X_train, y_train = oversample.fit_resample(X_train, y_train)

grid = {
    'estimator': base_models,
    'n_estimators': [200,500,1000]
}

EvaluateModel(X_train, y_train, X_test, y_test, model, grid)