import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from process_data import ProcessData

X_train,X_test,y_train,y_test = ProcessData()

model = AdaBoostClassifier(
            estimator=
            DecisionTreeClassifier(max_depth=1, criterion='entropy'),
            n_estimators=1000)
model.fit(X_train, y_train)
score = model.score(X_test, y_test.ravel())
pred = model.predict(X_test)

cm = metrics.confusion_matrix(y_test, pred)
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15)
plt.show()
