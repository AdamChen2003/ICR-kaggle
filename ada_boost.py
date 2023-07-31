from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from process_data import getBinaryClassData, splitTrainAndTest, getMultiClassData
from evaluate_model import EvaluateModel


X,y = getBinaryClassData()
# X = pca(X)
X_train, y_train, X_test, y_test = splitTrainAndTest(X, y)
model = AdaBoostClassifier(random_state=1)


base_models = []
for i in [1,2,3]:
    for criteria in ['gini', 'entropy', 'log_loss']:
        base_models.append(DecisionTreeClassifier(max_depth=i, criterion=criteria))
        base_models.append(DecisionTreeClassifier(max_depth=i, criterion=criteria,class_weight='balanced'))

grid = {
    'estimator': base_models,
    'n_estimators': [50,100,250,500]
}

base_models_os = []
for i in [1,2,3]:
    for criteria in ['gini', 'entropy', 'log_loss']:
        base_models_os.append(DecisionTreeClassifier(max_depth=i, criterion=criteria))

grid_os = {
    'classification__estimator': base_models_os,
    'classification__n_estimators': [50,100,250,500]
}

# No sampling
# EvaluateModel(X_train, y_train, X_test, y_test, model, grid, False)

# Oversampling
# EvaluateModel(X_train, y_train, X_test, y_test, model, grid_os, True)

X,y = getMultiClassData()
X_train, y_train, X_test, y_test = splitTrainAndTest(X, y)

# Multi class with no sampling
# EvaluateModel(X_train, y_train, X_test, y_test, model, grid, False, multi=True)

# Multi class with oversampling
EvaluateModel(X_train, y_train, X_test, y_test, model, grid_os, True, multi=True)