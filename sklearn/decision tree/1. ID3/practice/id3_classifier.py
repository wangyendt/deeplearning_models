#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author:  wangye
@file: id3_classifier.py 
@time: 2020/03/24
@contact: wangye.hope@gmail.com
@site:  
@software: PyCharm 
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def read_data_and_preprocessing():
    data = pd.read_excel('data.xlsx')
    # print(data.head())
    data.iloc[:, 1:] = data.iloc[:, 1:].apply(LabelEncoder().fit_transform)
    data = data.drop(labels=['编号', '好瓜', ], axis=1)
    target = data['好瓜']
    return data, target


def main():
    x, y = read_data_and_preprocessing()
    print(y)


if __name__ == '__main__':
    main()
