#!/usr/bin/env python
# encoding: utf-8

"""
@author: Wayne
@contact: wangye.hope@gmail.com
@software: PyCharm
@file: knc
@time: 2020/4/29 10:14
"""

from sklearn import datasets, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


def main():
    x, y = datasets.load_iris(True)
    x_tr, x_t, y_tr, y_t = train_test_split(x, y, test_size=0.3)
    k_score = []
    k_range = range(1, 31)
    for k in k_range:
        knc = KNeighborsClassifier(n_neighbors=k)
        # scores = cross_val_score(knc, x_tr, y_tr, cv=10, scoring='accuracy')
        scores = -cross_val_score(knc, x_tr, y_tr, cv=10, scoring='neg_mean_squared_error')
        # print(scores.mean())
        k_score.append(scores.mean())
    k_final = min(k_range, key=lambda t: k_score[t - 1])
    knc = KNeighborsClassifier(n_neighbors=k_final)
    knc.fit(x_tr, y_tr)
    print(f'k={k_final}, score={knc.score(x_t, y_t)}')
    plt.plot(k_score)
    plt.show()


if __name__ == '__main__':
    main()
