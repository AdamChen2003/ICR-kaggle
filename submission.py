import numpy as np
import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE

def balancedLogLossScorer(clf, X, y_true):
    y_pred = clf.predict_proba(X)
    return -balancedLogLoss(y_true, y_pred)

def balancedLogLoss(y_true, y_pred):
    # calculate the number of observations for each class
    N_0 = np.sum(1 - y_true)
    N_1 = np.sum(y_true)
    # calculate the weights for each class
    w_0 = 1 / N_0
    w_1 = 1 / N_1
    # calculate the predicted probabilities for each class
    p_0 = np.clip(y_pred[:, 0], 1e-15, 1 - 1e-15)
    p_1 = np.clip(y_pred[:, 1], 1e-15, 1 - 1e-15)
    # calculate the log loss for each class

    log_loss_0 = -w_0 * np.sum((1-y_true) * np.log(p_0))
    log_loss_1 = -w_1 * np.sum(y_true * np.log(p_1))
    # calculate the balanced logarithmic loss
    balanced_log_loss = (log_loss_0 + log_loss_1) / 2
    return balanced_log_loss

def PRE(val, data):
    t = len(data[(data['EJ'] == val) & (data['Class'] == 1)])
    f = len(data[(data['EJ'] == val) & (data['Class'] == 0)])
    if f:
        return t/f
    return 1

def getBinaryClassData():
    data = pd.read_csv('datasets/train.csv')
    test = pd.read_csv('datasets/test.csv')
    for i in data.columns[data.isnull().any(axis=0)]:
        data[i].fillna(data[i].mean(),inplace=True)

    for i in test.columns[test.isnull().any(axis=0)]:
        test[i].fillna(data[i].mean(),inplace=True)

    X_train = data.iloc[:,1:57]
    y_train = data.iloc[:,57]
    X_test = test.iloc[:,1:]
    PRE_A = PRE('A', data)
    PRE_B = PRE('B', data)
    X_test = X_test.replace('A', PRE_A)
    X_test = X_test.replace('B', PRE_B)
    X_train = X_train.replace('A', PRE_A)
    X_train = X_train.replace('B', PRE_B)
    
    return np.array(X_train), np.array(y_train).reshape(-1,1), np.array(X_test)

X_train, y_train, X_test = getBinaryClassData()
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
lr = LogisticRegression(C=0.1,class_weight='balanced',fit_intercept=False)
sfs = SequentialFeatureSelector(lr, n_features_to_select=50, scoring=balancedLogLossScorer)
sfs.fit(X_train, y_train.ravel())
X_train = sfs.transform(X_train)
X_test = sfs.transform(X_test)
model = AdaBoostClassifier(estimator=DecisionTreeClassifier(criterion='entropy', max_depth=3), n_estimators=500)
sm = SMOTE()
X_train, y_train = sm.fit_resample(X_train, y_train)
clf = model.fit(X_train, y_train.ravel())
predictions = clf.predict_proba(X_test)
print(predictions)
# sample_submission = pd.read_csv("/kaggle/input/icr-identify-age-related-conditions/sample_submission.csv")
# sample_submission[['class_0', 'class_1']] = predictions
# sample_submission.to_csv('/kaggle/working/submission.csv', index=False)