#!/usr/bin/env python
# encoding: utf-8

"""
@author: Wayne
@contact: wangye.hope@gmail.com
@software: PyCharm
@file: linear regression
@time: 2020/4/29 9:32
"""

from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


def main():
    x, y = datasets.make_regression(100, 1, noise=5)
    x = preprocessing.scale(x)
    y = preprocessing.scale(y)
    print(x.shape, y.shape)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    lr = LinearRegression(n_jobs=8)
    lr.fit(x_train, y_train)
    print(lr.coef_, lr.intercept_)
    print(lr.get_params())
    print(lr.score(x_test, y_test))


if __name__ == '__main__':
    main()
