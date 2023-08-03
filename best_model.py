from sklearn.metrics import f1_score
from evaluate_model import balancedLogLossScorer, balancedLogLoss
from process_data import Standardize, getBinaryClassData
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
import numpy as np

features = [5,10,15,20,25,30,35,40,45,50,56]
scores = {}

for n in features:
    scores[n] = {
        'accuracy': [],
        'f1 score': [],
        'balanced log loss': []
    }
for n in features:
    for _ in range(10):
        X, y = getBinaryClassData()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify=y, shuffle=True, random_state=1)
        X_train, X_test = Standardize(X_train, X_test)
        lr = LogisticRegression(C=0.1,class_weight='balanced',fit_intercept=False)
        if n != 56:
            sfs = SequentialFeatureSelector(lr, n_features_to_select=n, scoring=balancedLogLossScorer)
            sfs.fit(X_train, y_train.ravel())
            X_train = sfs.transform(X_train)
            X_test = sfs.transform(X_test)
            # model = RandomForestClassifier(criterion='log_loss', max_depth=7, n_estimators=800, random_state=42)
        # model = AdaBoostClassifier(estimator=DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=2), n_estimators=500)
        model = LogisticRegression(C=0.1,class_weight='balanced',fit_intercept=False)
        # sm = SMOTE(random_state=42)
        # X_train, y_train = sm.fit_resample(X_train, y_train)
        clf = model.fit(X_train, y_train.ravel())
        predictions = clf.predict(X_test)
        proba_predictions = clf.predict_proba(X_test)
        scores[n]['accuracy'].append(clf.score(X_test, y_test.ravel()))
        scores[n]['f1 score'].append(f1_score(predictions, y_test.ravel()))
        scores[n]['balanced log loss'].append(balancedLogLoss(y_test.ravel(), proba_predictions))

for n in scores:
    print(f'features: {n}')
    print(f'Accuracy mean: {np.mean(scores[n]["accuracy"])}')
    print(f'Accuracy std: {np.std(scores[n]["accuracy"])}')
    print(f'F1 score mean: {np.mean(scores[n]["f1 score"])}')
    print(f'F1 score std: {np.std(scores[n]["f1 score"])}')
    print(f'Balanced Log Loss mean: {np.mean(scores[n]["balanced log loss"])}')
    print(f'Balanced Log Loss std: {np.std(scores[n]["balanced log loss"])}')