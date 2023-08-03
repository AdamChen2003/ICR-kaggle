from evaluate_model import balancedLogLoss
from process_data import Standardize
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE

def PRE(val, data):
    t = len(data[(data['EJ'] == val) & (data['Class'] == 1)])
    f = len(data[(data['EJ'] == val) & (data['Class'] == 0)])
    if f:
        return t/f
    return 1

def getBinaryClassData():
    data = pd.read_csv('datasets/train.csv')
    for i in data.columns[data.isnull().any(axis=0)]:
        data[i].fillna(data[i].mean(),inplace=True)
    X = data.iloc[:,1:57]
    y = data.iloc[:,57]
    PRE_A = PRE('A', data)
    PRE_B = PRE('B', data)
    X = X.replace('A', PRE_A)
    X = X.replace('B', PRE_B)
    
    return np.array(X), np.array(y).reshape(-1,1), PRE_A, PRE_B

X, y, PRE_A, PRE_B= getBinaryClassData()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify=y, shuffle=True, random_state=1)
X_train, X_test = Standardize(X_train, X_test)
model = AdaBoostClassifier(estimator=DecisionTreeClassifier(criterion='entropy', max_depth=3), n_estimators=500)
sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)
clf = model.fit(X_train, y_train.ravel())
predictions = clf.predict(X_test)
proba_predictions = clf.predict_proba(X_test)
print(f'accuracy: {clf.score(X_test, y_test.ravel())}')
print(f'f1 score: {f1_score(predictions, y_test.ravel())}')
print(f'log loss: {log_loss(y_test.ravel(), proba_predictions[:,1])}')
print(f'balanced log loss: {balancedLogLoss(y_test.ravel(), proba_predictions)}')