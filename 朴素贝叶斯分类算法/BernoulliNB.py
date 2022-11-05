# -*- coding = utf-8 -*-
# Author：SEMHAQ
# Date:2022年11月04日
# Time:21:43
# File:BernoulliNB.py
# Software:PyCharm

from sklearn.model_selection import cross_val_score

# 导入鸢尾花数据集
from sklearn.datasets import load_iris

# 数据选取


iris_data = load_iris()['data']
iris_target = load_iris()['target']
# 数据预览
# print(iris_data)
# print(iris_target)

# 用伯努利模型进行预测和评估
from sklearn.naive_bayes import BernoulliNB

model = BernoulliNB()
result = model.fit(iris_data, iris_target)
predict = model.predict(iris_data)

# 计算错误个数
print("伯努利模型：")
print("数据总数：", len(iris_data), "  错误个数：", (iris_target != predict).sum())
# 模型评分
scores = cross_val_score(model, iris_data, iris_target)
print("Accuracy:%.3f" % scores.mean())

print(iris_target)
print(predict)

