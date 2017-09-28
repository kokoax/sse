#! coding: utf-8
import sys
import random
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import re
from sklearn import datasets
from sklearn.decomposition import PCA

class SSE:
    def __init__(self, cluster):
        self.nrow = 0
        self.ncol = 0
        self.data_flg = 0
        self.data_sets = self.getDataSets()
        self.plot_iris(cluster)

    def plot_iris(self, clusters):
        datasets = self.data_sets.data
        pca = PCA(n_components = 2)
        pca.fit(self.data_sets['data'])
        iris = pca.transform(self.data_sets.data)
        colors = ['red', 'blue', 'yellow']

        for (point,cluster) in zip(iris,clusters):
            plt.scatter(point[0], point[1], c=colors[cluster])

        plt.show()

    def getDataSets(self):
        if self.data_flg == 0:
            data_sets = datasets.load_iris()
        elif self.data_flg == 1:
            data_sets = datasets.load_digits()

        self.nrow, self.ncol = data_sets.data.shape

        return data_sets

cluster = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1,1,1,1,1,0,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,0,2,0,0,0,2,2,2,2,2,0,2,2,2,2,2,2,2,2,2,2,2,2,2,0,2,2,0,2,0,0,0,0,2,0,0,0,2,2,0,2,0,0,0,0,0,2,0,2,0,0,0,0,2,2,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0
    ,0,0,0,]
# cluster = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1]
sse = SSE(cluster)

