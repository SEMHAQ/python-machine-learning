# -*- coding = utf-8 -*-
# Author：SEMHAQ
# Date:2022年11月04日
# Time:22:10
# File:SVM.py
# Software:PyCharm

from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn import metrics
from sklearn.svm import SVR


def print_evaluate(true, predicted):
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    print('MAE:', mae)  # 均方误差
    print('MSE:', mse)  # 均方根误差
    print('RMSE:', rmse)  # 平均绝对误差
    print('R2 Square', r2_square)  # R方分数
    print('__________________________________')


# 加载数据集
boston = load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['target'] = boston.target
# 查看数据项
features = df[boston.feature_names]
target = df['target']
# 数据归一化处理
min_max_scaler = preprocessing.MinMaxScaler()
features = min_max_scaler.fit_transform(features)
# 数据集划分
split_num = int(len(features) * 0.8)
X_train = features[:split_num]
Y_train = target[:split_num]
X_test = features[split_num:]
Y_test = target[split_num:]
# 支持向量机建模
svm_reg = SVR(kernel='rbf', C=30, epsilon=0.01)

svm_reg.fit(X_train, Y_train)

test_predict = svm_reg.predict(X_test)
train_predict = svm_reg.predict(X_train)
print('Test set evaluation:\n_____________________________________')
print_evaluate(Y_test, test_predict)
print('Train set evaluation:\n_____________________________________')
print_evaluate(Y_train, train_predict)

# 将结果在指定路径保存为csv
# data -> 原始数据 ; result -> 预测数据 ; diff -> 原始数据与预测数据之差
df = pd.DataFrame({'data': Y_test, 'result': test_predict, 'diff': Y_test - test_predict})
df.to_csv(r'E:\python-machine-learning\支持向量机预测算法\result.csv', index=False)

# 可视化部分
# sns.set(font_scale=1.2)
# plt.rcParams['font.sans-serif'] = 'SimHei'
# plt.rcParams['axes.unicode_minus'] = False
# plt.rc('font', size=14)
# plt.plot(list(range(0, len(X_test))), Y_test, marker='o')
# plt.plot(list(range(0, len(X_test))), test_predict, marker='*')
# plt.legend(['真实值', '预测值'])
# plt.title('Boston房价支持向量机预测值与真实值的对比')
# plt.show()
