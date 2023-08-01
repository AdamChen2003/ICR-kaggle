from sklearn.naive_bayes import GaussianNB
from process_data import getBinaryClassData, splitTrainAndTest, getMultiClassData
from evaluate_model import EvaluateModel

X,y = getBinaryClassData()
# X = pca(X)
X_train, y_train, X_test, y_test = splitTrainAndTest(X, y)
model = GaussianNB()

# No sampling
EvaluateModel(X_train, y_train, X_test, y_test, model, {}, False)

# Oversampling
EvaluateModel(X_train, y_train, X_test, y_test, model, {}, True)

X,y = getMultiClassData()
X_train, y_train, X_test, y_test = splitTrainAndTest(X, y)

# Multi class with no sampling
EvaluateModel(X_train, y_train, X_test, y_test, model, {}, False, multi=True)

# Multi class with oversampling
EvaluateModel(X_train, y_train, X_test, y_test, model, {}, True, multi=True)