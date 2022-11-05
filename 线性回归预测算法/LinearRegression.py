# -*- coding = utf-8 -*-
# Author：SEMHAQ
# Date:2022年11月05日
# Time:18:45
# File:LinearRegression.py
# Software:PyCharm

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd

# 导入数据集
boston = datasets.load_boston()

# 取出自变量与因变量
x = boston.data
y = boston.target

# print(x)
# print(y)

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# 训练集训练
lr = LinearRegression()
lr.fit(x_train, y_train)

# 预测集预测
predict = lr.predict(x_test)
print(predict)

# 模型评分
score = r2_score(y_test, predict)
print(score)

# 将结果在指定路径保存为csv
df = pd.DataFrame({'result': predict})
df.to_csv(r'E:\python-machine-learning\线性回归预测算法\result.csv', index=False)
