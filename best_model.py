from evaluate_model import balancedLogLossScorer
from process_data import Standardize, getBinaryClassData
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression

X, y = getBinaryClassData()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify=y, shuffle=True, random_state=42)
X_train, X_test = Standardize(X_train, X_test)
lr = LogisticRegression(C=0.1,class_weight='balanced',fit_intercept=False)
sfs = SequentialFeatureSelector(lr, n_features_to_select=50, scoring=balancedLogLossScorer)
sfs.fit(X_train, y_train.ravel())
X_train = sfs.transform(X_train)
X_test = sfs.transform(X_test)
# model = RandomForestClassifier(criterion='log_loss', max_depth=7, n_estimators=800, random_state=42)
model = AdaBoostClassifier(estimator=DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42), n_estimators=500)
# model = LogisticRegression(C=0.1,class_weight='balanced',fit_intercept=False)
sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)
clf = model.fit(X_train, y_train.ravel())
predictions = clf.predict(X_test)
proba_predictions = clf.predict_proba(X_test)
print(proba_predictions)