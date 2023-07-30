import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, fbeta_score
from sklearn.model_selection import GridSearchCV

def EvaluateModel(X_train, y_train, X_test, y_test, model, grid):
    cv = GridSearchCV(estimator=model, param_grid=grid, cv= 5)
    cv.fit(X_train, y_train.ravel())
    score = cv.best_estimator_.score(X_test, y_test.ravel())
    print(f"Best parameters: {cv.best_params_}")
    print(f"best score on training set: {cv.best_score_}")
    print(f"best model accuracy on test set: {score}")
    print(f"f1 score: {f1_score(cv.best_estimator_.predict(X_test), y_test.ravel())}")
    print(f"f2 score: {fbeta_score(cv.best_estimator_.predict(X_test), y_test.ravel(), beta=2)}")

    plt.figure(figsize=(9,9))
    cm = confusion_matrix(y_test, cv.best_estimator_.predict(X_test))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Accuracy Score: {0}'.format(score)
    plt.title(all_sample_title, size = 15)
    # plt.show()
