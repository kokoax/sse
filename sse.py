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
    def __init__(self):
        self.nrow = 0
        self.ncol = 0
        self.data_flg = 0
        self.data_sets = self.getDataSets()
        self.sse = self.calc_sse()
        self.plot_iris()
        print("SSE: ", self.sse)

    def plot_iris(self):
        datasets = self.data_sets['data']
        pca = PCA(n_components = 2)
        pca.fit(self.data_sets['data'])
        iris = pca.transform(self.data_sets['data'])

        colors = ['red', 'blue', 'yellow']
        for point in iris:
            plt.plot(point[0], point[1], c=colors[cluster])

        plt.show()

    def get_centroid(self):
        return [sum(self.data_sets['data'][:,i])/self.nrow for i in range(self.ncol)]

    def calc_sse(self):
        centroid = self.get_centroid()
        sse = 0
        for data in self.data_sets['data']:
            sum_error = 0
            for i in range(self.ncol):
                sum_error = (data[i]-centroid[i]) ** 2
            sse += np.sqrt(sum_error)
        return sse

    def getDataSets(self):
        if self.data_flg == 0:
            data_sets = datasets.load_iris()
        elif self.data_flg == 1:
            data_sets = datasets.load_digits()

        self.all_data_sets = data_sets
        self.nrow, self.ncol = data_sets.data.shape
        tmp = 2

        data_sets['data'] = np.array([data_sets['data'][i+int(self.nrow/3*tmp)] for i in range(int(self.nrow/3))])
        # print(data_sets['data'])

        return data_sets

sse = SSE()

