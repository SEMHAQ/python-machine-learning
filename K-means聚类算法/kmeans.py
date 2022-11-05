# -*- coding = utf-8 -*-
# Author：SEMHAQ
# Date:2022年11月04日
# Time:21:39
# File:kmeans.py
# Software:PyCharm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

X = []
f = open('melon.txt', 'r')
for line in f:
    X.append(np.array(line.split(' '), dtype=np.string_).astype(np.float64))
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)

colors = ['red', 'green', 'blue']

for i, cluster in enumerate(kmeans.labels_):
    plt.scatter(X[i][0], X[i][1], color=colors[cluster])
plt.show()
