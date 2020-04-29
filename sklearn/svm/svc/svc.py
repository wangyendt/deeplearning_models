#!/usr/bin/env python
# encoding: utf-8

"""
@author: Wayne
@contact: wangye.hope@gmail.com
@software: PyCharm
@file: svc
@time: 2020/4/29 10:06
"""

from sklearn import datasets
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
import joblib
import numpy as np


def main():
    # x, y = datasets.make_classification(
    #     n_samples=500, n_features=2, n_redundant=0, n_informative=2,
    #     random_state=1, n_clusters_per_class=1, scale=100
    # )
    # x = preprocessing.scale(x)
    x, y = datasets.load_digits(return_X_y=True)
    # plt.scatter(x[:, 0], x[:, 1], c=y)
    # plt.show()
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    gamma_range = np.logspace(-6, -2.3, 5)
    train_loss, test_loss = validation_curve(
        SVC(), x_train, y_train, param_name='gamma', param_range=gamma_range, cv=10,
        scoring='neg_mean_squared_error'
    )
    # train_size, train_loss, test_loss = learning_curve(
    #     SVC(gamma=1000), x_train, y_train, cv=10,
    #     scoring='neg_mean_squared_error',
    #     train_sizes=np.arange(0.05, 1, 0.05)
    # )
    train_loss_mean = -np.mean(train_loss, axis=1)
    test_loss_mean = -np.mean(test_loss, axis=1)
    plt.plot(gamma_range, train_loss_mean, 'o-', color='r', label='Training')
    plt.plot(gamma_range, test_loss_mean, 'o-', color='g', label='Cross-Validation')
    # plt.plot(train_size, train_loss_mean, 'o-', color='r', label='Training')
    # plt.plot(train_size, test_loss_mean, 'o-', color='g', label='Cross-Validation')
    plt.legend()
    plt.show()
    svc = SVC(gamma=0.0005)
    svc.fit(x_train, y_train)
    joblib.dump(svc,'saved_model.pkl')
    svc_ = joblib.load('saved_model.pkl')
    print(svc.score(x_test, y_test))
    print(svc_.score(x_test, y_test))


if __name__ == '__main__':
    main()
