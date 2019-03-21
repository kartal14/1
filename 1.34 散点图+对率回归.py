#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2019/3/7 20:29
#@Author: KartalLee
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
# import matplotlib.pylab as pl
data=np.loadtxt('watermelon_3a.csv',delimiter=",")#delimiter 间隔为","
print(data)
x=data[:,1:3]
y=data[:,-1]

f1 = plt.figure(0)
plt.title('watermelon_3a')
plt.xlabel('density')
plt.ylabel('ratio_sugar')
plt.scatter(x[y == 0,0], x[y == 0,1], marker = 'x', color = 'r', s=100, label = 'bad')
plt.scatter(x[y == 1,0], x[y == 1,1], marker = 'o', color = 'g', s=100, label = 'good')
#scatter()函数的前面两个分别y为0或1的横纵坐标
plt.legend(loc = 'upper right')
# plt.show()



# generalization of test and train set
X_train, X_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.5, random_state=0)

# model training
log_model = LogisticRegression()
log_model.fit(X_train, y_train)

# model testing
y_pred = log_model.predict(X_test)

# summarize the accuracy of fitting
print(metrics.confusion_matrix(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))
precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_pred)
f2 = plt.figure(2)
h = 0.001
x0_min, x0_max = x[:, 0].min()-0.1, x[:, 0].max()+0.1
x1_min, x1_max = x[:, 1].min()-0.1, x[:, 1].max()+0.1
x0, x1 = np.meshgrid(np.arange(x0_min, x0_max, h),
                     np.arange(x1_min, x1_max, h))

# here "model" is your model's prediction (classification) function
z = log_model.predict(np.c_[x0.ravel(), x1.ravel()])


# Put the result into a color plot
z = z.reshape(x0.shape)
plt.contourf(x0, x1, z, cmap = plt.cm.Paired)

# Plot also the training pointsplt.title('watermelon_3a')
plt.title('watermelon_3a')
plt.xlabel('density')
plt.ylabel('ratio_sugar')
plt.scatter(x[y == 0,0], x[y == 0,1], marker = 'x', color = 'r', s=100, label = 'bad')
plt.scatter(x[y == 1,0], x[y == 1,1], marker = 'o', color = 'g', s=100, label = 'good')
plt.legend(loc = 'upper right')
plt.show()

