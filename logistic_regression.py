from sklearn.linear_model import LogisticRegression
from process_data import getBinaryClassData, getBinaryClassWeights
from evaluate_model import EvaluateModel

X,y = getBinaryClassData()
model = LogisticRegression(class_weight=getBinaryClassWeights())
EvaluateModel(X, y, model)