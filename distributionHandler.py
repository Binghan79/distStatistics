#!/usr/bin/env Python
# -*- coding:utf-8 -*-
# author: Binghan

import numpy as np
import pandas as pd
from scipy import stats, integrate
import matplotlib.pyplot as plt
import seaborn as sns


"""
    模块功能：
        绘制概率分布图
        估计概率密度函数
        从给定的概率密度函数中进行抽样
"""


class Distribution(object):
    """ 概率分布类 """
    def __init__(self, data):
        self.data = data
        self.dist = None

    def gaussian(self):
        """ 对data进行高斯核无参估计 """
        self.dist = stats.gaussian_kde(self.data)
        return self.dist

    def resample(self, size=1):
        return self.dist.resample(size)


def distributionHist(data, figPath=None):
    """ 直接绘图 """
    sns.distplot(data)
    if figPath is None:
        plt.show()
    else:
        plt.savefig(figPath)

if __name__ == "__main__":
    testDistribution = np.random.normal(size=100)
    testObj = Distribution(testDistribution)
    testObj.gaussian()
    samples = testObj.resample(10)
    print(samples)
    distributionHist(testDistribution)
