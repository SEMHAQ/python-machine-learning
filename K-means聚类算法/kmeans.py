# -*- coding = utf-8 -*-
# Author：SEMHAQ
# Date:2022年11月04日
# Time:21:39
# File:kmeans.py
# Software:PyCharm

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets
import pandas as pd

# 直接从sklearn中获取数据集
from sklearn.datasets import load_iris

X = load_iris()['data']
iris_target = load_iris()['target']


# 取前两个维度（萼片长度、萼片宽度），绘制数据分布图
# plt.scatter(X[:, 0], X[:, 1], c="red", marker='o', label='see')
# plt.xlabel('sepal length')
# plt.ylabel('sepal width')
# plt.legend(loc=2)
# plt.show()

def Model(n_clusters):
    estimator = KMeans(n_clusters=n_clusters)  # 构造聚类器
    return estimator


def train(estimator):
    estimator.fit(X)  # 聚类


# 初始化实例，并开启训练拟合
estimator = Model(3)
train(estimator)

label_pred = estimator.labels_  # 获取聚类标签
# 绘制k-means结果
x0 = X[label_pred == 0]
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]

print(label_pred)
print(iris_target)
print("数据总数：", len(X), "  错误个数：", (label_pred != iris_target).sum())


df = pd.DataFrame(
    {'Column_1': X[:, 0], 'Column_2': X[:, 1], 'Column_3': X[:, 2], 'Column_4': X[:, 3], 'data': iris_target,
     'result': label_pred, 'diff': label_pred == iris_target})
df.to_csv(r'E:\python-machine-learning\K-means聚类算法\result.csv', index=False)

# plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='label0')
# plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label1')
# plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='+', label='label2')
# plt.xlabel('sepal length')
# plt.ylabel('sepal width')
# plt.legend(loc=2)
# plt.show()
