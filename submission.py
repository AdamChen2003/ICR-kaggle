import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
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

X_train, y_train, PRE_A, PRE_B= getBinaryClassData()
X_train = StandardScaler().fit_transform(X_train)
model = AdaBoostClassifier(estimator=DecisionTreeClassifier(criterion='entropy', max_depth=3), n_estimators=500)
sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)
clf = model.fit(X_train, y_train.ravel())
data = pd.read_csv('datasets/test.csv')

for i in data.columns[data.isnull().any(axis=0)]:
    data[i].fillna(data[i].mean(),inplace=True)
X = data.iloc[:,1:57]
X = X.replace('A', PRE_A)
X = X.replace('B', PRE_B)
X = StandardScaler().fit_transform(X)
data = pd.read_csv('/kaggle/input/icr-identify-age-related-conditions/test.csv')
for i in data.columns[data.isnull().any(axis=0)]:
    data[i].fillna(data[i].mean(),inplace=True)
data = data.drop(['EJ'], axis=1)
X = data.iloc[:,1:]
X = StandardScaler().fit_transform(X)
predictions = clf.predict_proba(X)
sample_submission = pd.read_csv("/kaggle/input/icr-identify-age-related-conditions/sample_submission.csv")
sample_submission[['class_0', 'class_1']] = predictions
sample_submission.to_csv('/kaggle/working/submission.csv', index=False)