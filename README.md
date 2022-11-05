# Python机器学习简单应用

#### 介绍
利用sklearn机器学习库进行分类与预测。该仓库提出了两种预测算法和两种分类算法，预测算法方面，利用了 **神经网络及支持向量机（SVM）** ；分类算法方面，利用了 **K-Means聚类算法及朴素贝叶斯分类算法** 。

#### 软件架构
需要用到的module有： **sklearn、pandas、numpy、matplotlib、seaborn、sys** 。  

#### 使用说明
sklearn中文社区：https://scikit-learn.org.cn/  
波士顿房价：http://lib.stat.cmu.edu/datasets/boston  
所使用的到的数据集： **1、鸢尾花数据集；2、波士顿房价数据集。** 

#### 波士顿房价数据集介绍  

CRIM:城镇人均犯罪率  
ZN:住宅用地超过25000 sq.ft.的比例  
INDUS:城镇非零售商用土地的比例  
CHAS:边界是河流为1,否则0  
NOX: 一氧化氮浓度  
RM:住宅平均房间数  
AGE: 1940年之前建成的自用房屋比例  
DIS:到波士顿5个中心区域的加权距离  
RAD:辐射性公路的靠近指数  
TAX:每10000美元的全值财产税率  
PTRATIO:城镇师生比例  
LSTAT:人口中地位低下者的比例  
MEDV:自住房的平均房价，单位:千美元  

 The Boston house-price data of Harrison, D. and Rubinfeld, D.L. 'Hedonic  
 prices and the demand for clean air', J. Environ. Economics & Management,  
 vol.5, 81-102, 1978.   Used in Belsley, Kuh & Welsch, 'Regression diagnostics  
 ...', Wiley, 1980.   N.B. Various transformations are used in the table on  
 pages 244-261 of the latter.  

 Variables in order:  
 CRIM         per capita crime rate by town  
 ZN           proportion of residential land zoned for lots over 25,000 sq.ft.  
 INDUS        proportion of non-retail business acres per town  
 CHAS         Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)  
 NOX          nitric oxides concentration (parts per 10 million)  
 RM           average number of rooms per dwelling   
 AGE          proportion of owner-occupied units built prior to 1940  
 DIS          weighted distances to five Boston employment centres  
 RAD          index of accessibility to radial highways  
 TAX          full-value property-tax rate per 10,000 dollars  
 PTRATIO      pupil-teacher ratio by town  
 B            1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town  
 LSTAT        % lower status of the population  
 MEDV         Median value of owner-occupied homes in $1000's  


