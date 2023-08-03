import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, fbeta_score, roc_auc_score, log_loss, brier_score_loss, accuracy_score
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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

def EvaluateModel(X_train, y_train, X_test, y_test, model, grid, oversampling, multi=False):
    if oversampling:
        model = Pipeline([
            ('sampling', SMOTE(random_state=42)),
            ('sampling', RandomOverSampler(random_state=42)),
            ('scaler', StandardScaler()),
            # ('scaler', MinMaxScaler()),
            ('classification', model),
        ])
    else:
        model = Pipeline([
            ('scaler', StandardScaler()),
            # ('scaler', MinMaxScaler()),
            ('classification', model)
        ])
    if multi:
        cv = GridSearchCV(model, grid, cv=5, scoring='f1_micro').fit(X_train, y_train.ravel())
    else:
        # cv = GridSearchCV(model, grid, cv=ss, scoring='f1').fit(X_train, y_train.ravel())
        cv = GridSearchCV(model, grid, cv=5, scoring=balancedLogLossScorer).fit(X_train, y_train.ravel())
    score = cv.best_estimator_.score(X_test, y_test.ravel())
    predictions = cv.best_estimator_.predict(X_test)
    
    if multi:
        predictions[predictions > 0] = 1
        y_test[y_test > 0] = 1
        score = accuracy_score(y_test, predictions)

    proba_predictions = cv.best_estimator_.predict_proba(X_test)
    if multi:
        proba_predictions = np.array([[a, b + c + d] for a, b, c, d in proba_predictions])
    
    print(f'Best parameters: {cv.best_params_}')
    print(f'accuracy: {score}')
    print(f'f1 score: {f1_score(predictions, y_test.ravel())}')
    print(f'log loss: {log_loss(y_test.ravel(), proba_predictions[:,1])}')
    print(f'balanced log loss: {balancedLogLoss(y_test.ravel(), proba_predictions)}')
    # print(cv.cv_results_)