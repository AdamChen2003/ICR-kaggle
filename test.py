import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from process_data import getBinaryClassData, splitTrainAndTest
from math import log

X,y = getBinaryClassData()
# X = pca(X)
X_train, y_train, X_test, y_test = splitTrainAndTest(X, y)
model = LogisticRegression(random_state=1)
model.fit(X_train, y_train.ravel())

# def balancedLogLoss(y_true, y_pred):
#     print(y_pred)
#     N0, N1 = np.bincount(y_true)
#     class0 = np.array(np.where(y_true == 0)).ravel()
#     sum0 = 0
#     for i in class0:
#         y = 0
#         if y_true[i] == 0:
#             y = 1
#         sum0 += y * log(y_pred[i][0])
    
#     class1 = np.array(np.where(y_true == 1)).ravel()
#     sum1 = 0
#     for i in class1:
#         y = 0
#         if y_true[i] == 1:
#             y = 1
#         sum1 += y * log(y_pred[i][1])
    
#     return -(sum0/N0 + sum1/N1)/2
    
# print(balancedLogLoss(y_test.ravel(), model.predict_proba(X_test)))
print(model.predict_proba(X_test))