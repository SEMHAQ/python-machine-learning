# -*- coding = utf-8 -*-
# Author：SEMHAQ
# Date:2022年11月04日
# Time:21:42
# File:GaussianNB.py
# Software:PyCharm
# 导入鸢尾花数据集
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
import pandas as pd
# 数据选取
iris_data = load_iris()['data']
iris_target = load_iris()['target']

# 数据预览
# print(iris_data)
# print(iris_target)

# 用高斯模型进行预测并评估
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
result = model.fit(iris_data, iris_target)
predict = model.predict(iris_data)

# 对预测结果的正确个数进行计算
print("高斯模型：")
print("数据总数：", len(iris_data), "  错误个数：", (iris_target != predict).sum())
# 对模型进行评估
scores = cross_val_score(model, iris_data, iris_target)
print("Accuracy:%.3f" % scores.mean())

print(iris_target)
print(predict)

df = pd.DataFrame({'data': iris_target, 'result': predict, 'diff': iris_target == predict})
df.to_csv(r'E:\python-machine-learning\朴素贝叶斯分类算法\GaussianNB_result.csv', index=False)
