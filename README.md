# Python机器学习简单应用

#### 介绍
利用sklearn机器学习库进行分类与预测。该仓库提出了两种预测算法和两种分类算法，预测算法方面，利用了 **神经网络及支持向量机（SVM）**；分类算法方面，利用了 **K-Means聚类算法及朴素贝叶斯分类算法** 。

#### 项目架构
需要用到的module有： **sklearn、pandas、numpy、matplotlib、seaborn、sys** 。  

#### 使用说明
sklearn中文社区：https://scikit-learn.org.cn/  
波士顿房价：http://lib.stat.cmu.edu/datasets/boston  
所使用的到的数据集： **1、波士顿房价数据集（预测）；2、鸢尾花数据集（分类）。** 

#### 波士顿房价数据集介绍  
| 指标  | 解释  |
|---|---|
| CRIM  |  城镇人均犯罪率 |
| ZN  |  住宅用地超过25000 sq.ft.的比例   |
| INDUS  | 城镇非零售商用土地的比例  |
|  CHAS | 边界是河流为1,否则0    |
| NOX  | 一氧化氮浓度    |
| RM  | 住宅平均房间数    |
| AGE  | 1940年之前建成的自用房屋比例    |
|  DIS | 到波士顿5个中心区域的加权距离   |
|  RAD |  辐射性公路的靠近指数 |
|  TAX | 每10000美元的全值财产税率    |
|  PTRATIO |  城镇师生比例   |
|  LSTAT | 人口中地位低下者的比例    |
|  MEDV | 自住房的平均房价，单位:千美元   |

#### 鸢尾花数据集介绍  
iris_training.csv训练数据集，120条样本数据；iris_test.csv测试数据集，30条数据。  
| 指标  | 解释  |
|---|---|
| Sepal Length  |  花萼长度 |
| Sepal Width |  花萼宽度  |
| Petal Length | 花瓣长度  |
|  Petal Width | 花瓣宽度    |
| 0  | 山鸢尾（Setosa）    |
| 1  | 变色鸢尾（Versicolor）|
|2 | 维吉尼亚鸢尾（Virginical）    |





