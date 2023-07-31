import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, fbeta_score, roc_auc_score, log_loss, brier_score_loss, make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

def EvaluateModel(X_train, y_train, X_test, y_test, model, grid, oversampling, multi=False):
    if oversampling:
        model = Pipeline([
            ('sampling', SMOTE()),
            ('classification', model)
        ])
    # f1 = make_scorer(f1_score)
    cv = GridSearchCV(estimator=model, param_grid=grid, cv=5, scoring='neg_log_loss')
    cv.fit(X_train, y_train.ravel())
    score = cv.best_estimator_.score(X_test, y_test.ravel())
    predictions = cv.best_estimator_.predict(X_test)
    
    if multi:
        predictions[predictions > 0] = 1
        y_test[y_test > 0] = 1
        score = accuracy_score(y_test, predictions)

    print(f'Best parameters: {cv.best_params_}')
    print(f'accuracy: {score}')
    print(f'f1 score: {f1_score(predictions, y_test.ravel())}')
    print(f'f2 score: {fbeta_score(predictions, y_test.ravel(), beta=2)}')
    print(f'ROC AUC score: {roc_auc_score(predictions, y_test.ravel())}')
    print(f'log loss: {log_loss(predictions, y_test.ravel())}')
    print(f'brier score: {brier_score_loss(predictions, y_test.ravel())}')

    plt.figure(figsize=(9,9))
    cm = confusion_matrix(y_test, cv.best_estimator_.predict(X_test))
    sns.heatmap(cm, annot=True, fmt='.3f', linewidths=.5, square = True, cmap = 'Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Accuracy Score: {0}'.format(score)
    plt.title(all_sample_title, size = 15)
    # plt.show()