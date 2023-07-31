from sklearn.naive_bayes import GaussianNB
from process_data import getBinaryClassData, splitTrainAndTest
from evaluate_model import EvaluateModel

X,y = getBinaryClassData()
# X = pca(X)
X_train, y_train, X_test, y_test = splitTrainAndTest(X, y)
model = GaussianNB()

# No sampling
EvaluateModel(X_train, y_train, X_test, y_test, model, {}, False)

# Oversampling
EvaluateModel(X_train, y_train, X_test, y_test, model, {}, True)