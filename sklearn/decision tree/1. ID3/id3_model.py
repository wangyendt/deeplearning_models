#!/usr/bin/env python
# encoding: utf-8

"""
@author: Wayne
@contact: wangye.hope@gmail.com
@software: PyCharm
@file: id3_model
@time: 2020/6/4 13:49
"""

import collections
import math


def calc_entropy(n_classes: list) -> float:
    n = sum(n_classes)
    return -sum(k / n * math.log2(k / n) for k in n_classes)


def calc_id3(title: list, data: list(list()), use_features: list):
    D = [d[-1] for d in data]
    H_D = calc_entropy(list(collections.Counter(D).values()))
    use_features_indices = [title.index(uf) for uf in use_features]
    H_Di = collections.defaultdict(lambda: H_D)
    for ufi in use_features_indices:
        Di = collections.defaultdict(collections.Counter)
        for d in data:
            Di[d[ufi]] += collections.Counter([d[-1]])
        freq = {di: sum(Di[di].values()) for di in Di.keys()}
        for k in Di.keys():
            H_Di[title[ufi]] -= calc_entropy(list(Di[k].values())) * freq[k] / sum(freq.values())
    print(H_Di)


if __name__ == '__main__':
    title = ['姓名', '年龄', '长相', '身高', '写代码', '是否见面']
    data = [['小A', '老', '帅', '高', '不会', '不见'],
            ['小B', '年轻', '一般', '中等', '会', '见'],
            ['小C', '年轻', '丑', '高', '不会', '不见'],
            ['小D', '年轻', '一般', '高', '会', '见'],
            ['小E', '年轻', '一般', '低', '不会', '不见']]
    use_features = ['年龄', '长相', '身高', '写代码']
    calc_id3(title, data, use_features)
