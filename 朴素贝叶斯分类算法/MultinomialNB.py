# -*- coding = utf-8 -*-
# Author：SEMHAQ
# Date:2022年11月04日
# Time:21:42
# File:MultinomialNB.py
# Software:PyCharm
import pandas as pd
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

# 用多项式建立模型进行预测和评估
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
result = model.fit(iris_data, iris_target)
# 预测
predict = result.predict(iris_data)
# 计算错误个数
print("多项式模型：")
print("数据总数：", iris_data.shape[0], "  错误个数：", (iris_target != predict).sum())
# 模型评分
scores = cross_val_score(model, iris_data, iris_target)
print("Accuracy:%.3f" % scores.mean())

print(iris_target)
print(predict)

df = pd.DataFrame({'data': iris_target, 'result': predict, 'diff': iris_target == predict})
df.to_csv(r'E:\python-machine-learning\朴素贝叶斯分类算法\MultinomialNB_result.csv', index=False)

