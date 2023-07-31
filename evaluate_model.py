from math import log
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, fbeta_score, roc_auc_score, log_loss, brier_score_loss, make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

def balancedLogLoss(y_true, y_pred):
    N0, N1 = np.bincount(y_true)
    class0 = np.array(np.where(y_true == 0)).ravel()
    sum0 = 0
    for i in class0:
        if y_true[i] == 0:
            p = max(min(y_pred[i], 1-10**(-15)),10**(-15))
            sum0 += log(1-p)
    
    class1 = np.array(np.where(y_true == 1)).ravel()
    sum1 = 0
    for i in class1:
        if y_true[i] == 1:
            p = max(min(y_pred[i], 1-10**(-15)),10**(-15))
            sum1 += log(p)
    
    return -(sum0/N0 + sum1/N1)/2

def EvaluateModel(X_train, y_train, X_test, y_test, model, grid, oversampling, multi=False):
    if oversampling:
        model = Pipeline([
            ('sampling', SMOTE()),
            ('classification', model)
        ])
    # f1 = make_scorer(f1_score)
    
    cv = GridSearchCV(estimator=model, param_grid=grid, cv=5, scoring=make_scorer(balancedLogLoss, needs_proba=True, greater_is_better=False))
    # cv = GridSearchCV(estimator=model, param_grid=grid, cv=5, scoring='neg_log_loss')
    cv.fit(X_train, y_train.ravel())
    score = cv.best_estimator_.score(X_test, y_test.ravel())
    predictions = cv.best_estimator_.predict(X_test)
    
    if multi:
        predictions[predictions > 0] = 1
        y_test[y_test > 0] = 1
        score = accuracy_score(y_test, predictions)

    proba_predictions = cv.best_estimator_.predict_proba(X_test)
    print(f'Best parameters: {cv.best_params_}')
    print(f'accuracy: {score}')
    print(f'f1 score: {f1_score(predictions, y_test.ravel())}')
    print(f'f2 score: {fbeta_score(predictions, y_test.ravel(), beta=2)}')
    print(f'ROC AUC score: {roc_auc_score(predictions, y_test.ravel())}')
    print(f'log loss: {log_loss(y_test.ravel(), proba_predictions[:,1])}')
    print(f'brier score: {log_loss(y_test.ravel(), proba_predictions[:,1])}')
    print(f'balanced log loss: {balancedLogLoss(y_test.ravel(), proba_predictions[:,1])}')

    plt.figure(figsize=(9,9))
    cm = confusion_matrix(y_test, cv.best_estimator_.predict(X_test))
    sns.heatmap(cm, annot=True, fmt='.3f', linewidths=.5, square = True, cmap = 'Blues_r')
    plt.ylabel('Actual label'),
    plt.xlabel('Predicted label')
    all_sample_title = 'Accuracy Score: {0}'.format(score)
    plt.title(all_sample_title, size = 15)
    # plt.show()