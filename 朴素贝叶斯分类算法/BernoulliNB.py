# -*- coding = utf-8 -*-
# Author：SEMHAQ
# Date:2022年11月04日
# Time:21:43
# File:BernoulliNB.py
# Software:PyCharm
import pandas as pd
import numpy as np

X = np.array([[1.14, 1.78], [1.18, 1.96], [1.20, 1.86], [1.26, 2.00], [1.28, 2.00],
              [1.30, 1.96], [1.24, 1.72], [1.36, 1.74], [1.38, 1.64], [1.38, 1.82],
              [1.38, 1.90], [1.40, 1.70], [1.48, 1.82], [1.54, 1.82], [1.56, 2.08]])
Y = np.hstack((np.ones(6), np.ones(9) * 2))  # 数组合并

from sklearn.naive_bayes import BernoulliNB

model = BernoulliNB()
model.fit(X, Y)

print("预测结果")
print(model.predict([[1.24, 1.80]]))
print("样本为1类的概率")
print(model.predict_proba([[1.24, 1.80]]))
print("样本为2类的概率")
print(model.predict_log_proba([[1.24, 1.80]]))

print("预测结果")
print(model.predict([[1.29, 1.81], [1.43, 2.03]]))
