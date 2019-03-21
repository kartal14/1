import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
import numpy as np
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
iris=datasets.load_iris()
log_model = LogisticRegression()
scores = cross_val_score(log_model, iris.data, iris.target, cv=10)
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
accuracy = 0
for train, test in loo.split(iris.data):
    log_model.fit(iris.data[train], iris.target[train])  # fitting
    y_p = log_model.predict(iris.data[test])
    if y_p == iris.target[test] : accuracy += 1
print(accuracy / np.shape(iris.data)[0])
