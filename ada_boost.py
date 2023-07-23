from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from process_data import getBinaryClassData, getBinaryClassWeights, getMultiClassData, getMultiClassWeights
from evaluate_model import EvaluateModel

X, y = getBinaryClassData()
model = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1, criterion='entropy',class_weight=getBinaryClassWeights()),n_estimators=1000)
EvaluateModel(X, y, model)

# X,y = getMultiClassData()
# model = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1, criterion='entropy',class_weight=getMultiClassWeights()),n_estimators=1000)
# EvaluateModel(X, y, model, True)