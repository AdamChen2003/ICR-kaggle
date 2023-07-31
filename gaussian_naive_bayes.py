import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from process_data import getBinaryClassData, Normalize, pca
from evaluate_model import EvaluateModel
from imblearn.over_sampling import SMOTE

X,y = getBinaryClassData()
# X = pca(X)
X_train, y_train, X_test, y_test = splitTrainAndTest(X, y)
model = GaussianNB()

# No sampling
EvaluateModel(X_train, y_train, X_test, y_test, model, {}, False)

# Oversampling
EvaluateModel(X_train, y_train, X_test, y_test, model, {}, True)