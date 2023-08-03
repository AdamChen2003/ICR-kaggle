from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from process_data import getBinaryClassData, getMultiClassData
from evaluate_model import EvaluateModel

X,y = getBinaryClassData()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify=y, shuffle=True, random_state=1)
model = GaussianNB()

# No sampling
EvaluateModel(X_train, y_train, X_test, y_test, model, {}, False)

# Oversampling
EvaluateModel(X_train, y_train, X_test, y_test, model, {}, True)

X,y = getMultiClassData()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify=y, shuffle=True, random_state=1)

# Multi class with no sampling
EvaluateModel(X_train, y_train, X_test, y_test, model, {}, False, multi=True)

# Multi class with oversampling
EvaluateModel(X_train, y_train, X_test, y_test, model, {}, True, multi=True)