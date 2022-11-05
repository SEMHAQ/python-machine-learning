# -*- coding = utf-8 -*-
# Author：SEMHAQ
# Date:2022年11月04日
# Time:21:43
# File:BernoulliNB.py
# Software:PyCharm

# 导入鸢尾花数据集
from sklearn.datasets import load_iris

# 数据选取
iris_data = load_iris()['data']
iris_target = load_iris()['target']

# 用高斯模型进行预测并评估
from sklearn.naive_bayes import GaussianNB
mol = GaussianNB()
result = mol.fit(iris_data,iris_target)
pred1 = mol.predict(iris_data)
# 对模型进行评估
from sklearn.model_selection import cross_val_score
scores = cross_val_score(mol,iris_data,iris_target,cv=10)
# 对预测结果的正确个数进行计算
print("高斯模型：")
print("数据总数：",len(iris_data),"  错误个数：",(iris_target != pred1).sum())
print("Accuracy:%.3f"%scores.mean())

# 用贝努里模型进行预测和评估
from sklearn.naive_bayes import BernoulliNB
bnb = BernoulliNB()
result2 = bnb.fit(iris_data,iris_target)
pred2 = bnb.predict(iris_data)
# 计算错误个数
print("贝努里模型：")
print("数据总数：",len(iris_data),"  错误个数：",(iris_target != pred2).sum())
#模型评分
scores2 = cross_val_score(bnb,iris_data,iris_target)
print("Accuracy:%.3f"%scores2.mean())

# 用多项式建立模型进行预测和评估
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
result3 = mnb.fit(iris_data,iris_target)
# 预测
pred3 = result3.predict(iris_data)
# 计算错误个数
print("多项式模型：")
print("数据总数：",iris_data.shape[0],"  错误个数：",(iris_target != pred3).sum())
# 模型评分
scores3 = cross_val_score(mnb,iris_data,iris_target)
print("Accuracy:%.3f"%scores3.mean())


