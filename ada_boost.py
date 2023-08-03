from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from process_data import getBinaryClassData, getMultiClassData
from evaluate_model import EvaluateModel

X,y = getBinaryClassData()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify=y, shuffle=True, random_state=1)
model = AdaBoostClassifier(random_state=1, estimator=DecisionTreeClassifier(max_depth=3, criterion='entropy', random_state=1))

base_models = []
for i in [1,3,5]:
    for j in ['gini', 'entropy', 'log_loss']:
        base_models.append(DecisionTreeClassifier(max_depth=i, criterion=j, random_state=1))
        base_models.append(DecisionTreeClassifier(max_depth=i, criterion=j, class_weight='balanced', random_state=1))

grid = {
    'classification__estimator': base_models,
    'classification__n_estimators': [250,500,1000]
}

base_models_os = []
for i in [1,3,5]:
        for j in ['gini', 'entropy', 'log_loss']:
            base_models_os.append(DecisionTreeClassifier(max_depth=i, criterion=j, random_state=1))

grid_os = {
    'classification__estimator': base_models_os,
    'classification__n_estimators': [250,500,100]
}

# No sampling
EvaluateModel(X_train, y_train, X_test, y_test, model, grid, False)

# Oversampling
EvaluateModel(X_train, y_train, X_test, y_test, model, grid_os, True)

X,y = getMultiClassData()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify=y, shuffle=True, random_state=1)

# Multi class with no sampling
EvaluateModel(X_train, y_train, X_test, y_test, model, grid, False, multi=True)

# Multi class with oversampling
EvaluateModel(X_train, y_train, X_test, y_test, model, grid_os, True, multi=True)