#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author:  wangye
@file: iris_preprocessing.py 
@time: 2020/03/24
@contact: wangye.hope@gmail.com
@site:  
@software: PyCharm 
"""
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import scale, StandardScaler, \
    minmax_scale, MinMaxScaler, \
    maxabs_scale, MaxAbsScaler, \
    robust_scale, RobustScaler, \
    normalize, Normalizer, \
    binarize, Binarizer, \
    OneHotEncoder, LabelEncoder, PolynomialFeatures, Imputer, FunctionTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score, permutation_test_score
from sklearn.metrics import make_scorer, accuracy_score, classification_report
from sklearn.dummy import DummyClassifier


def preprocessing(method: str, input: np.ndarray) -> np.ndarray:
    output = None
    if method == 'scale':
        output = scale(input)
    elif method == 'StandardScaler':
        output = StandardScaler().fit_transform(input)
    elif method == 'minmax_scale':
        output = minmax_scale(input)
    elif method == 'MinMaxScaler':
        output = MinMaxScaler().fit_transform(input)
    elif method == 'maxabs_scale':
        output = maxabs_scale(input)
    elif method == 'MaxAbsScaler':
        output = MaxAbsScaler().fit_transform(input)
    elif method == 'robust_scale':
        output = robust_scale(input)
    elif method == 'RobustScaler':
        output = RobustScaler().fit_transform(input)
    elif method == 'normalize':
        output = normalize(input)
    elif method == 'Normalizer':
        output = Normalizer().fit_transform(input)
    elif method == 'binarize':
        output = binarize(input)
    elif method == 'Binarizer':
        output = Binarizer().fit_transform(input)
    return output


def transform(input: np.ndarray) -> np.ndarray:
    return FunctionTransformer(np.log1p).transform(input)


def main():
    iris = datasets.load_iris()
    x, y = iris.data, iris.target
    print(x.shape, y.shape)
    scale_x = preprocessing('robust_scale', x)
    pipeline = Pipeline([
        ('scale', StandardScaler()),
        ('pca', PCA(n_components=2)),
        ('clf', LogisticRegression(random_state=0))
    ])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, stratify=y)
    print(x_train.shape, y_train.shape)
    pipeline.fit(x_train, y_train)
    param_grid = {
        'pca__n_components': [2, 3, 4],
        'clf__penalty': ['l1', 'l2']
    }
    grid = GridSearchCV(pipeline, param_grid, scoring='f1_weighted', n_jobs=1)
    grid.fit(x_train, y_train)
    print(grid.best_estimator_)
    print(grid.best_score_)

    # plt.plot(np.sort(x[:,0]))
    # plt.plot(np.sort(scale_x[:,0]))
    # plt.show()
    print(scale_x.shape)


if __name__ == '__main__':
    main()
