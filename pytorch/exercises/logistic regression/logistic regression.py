#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author:  wangye
@file: logistic regression.py 
@time: 2020/03/16
@contact: wangye.hope@gmail.com
@site:  
@software: PyCharm 
"""

import numpy as np
import torch
if __name__ == '__main__':
    data = np.loadtxt('german.data-numeric')
    print(data[0])
    print(data.shape)