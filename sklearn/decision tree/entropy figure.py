#!/usr/bin/env python
# encoding: utf-8

"""
@author: Wayne
@contact: wangye.hope@gmail.com
@software: PyCharm
@file: entropy figure
@time: 2020/4/29 17:32
"""

import numpy as np
import matplotlib.pyplot as plt


def entropy(input):
    if input == 0 or input == 1:
        return 0
    else:
        return -input * np.log2(input) - (1 - input) * np.log2(1 - input)


def main():
    x = np.linspace(0, 100, 101) / 100
    y = list(map(entropy, x))
    plt.plot(x, y)
    plt.title('Entropy')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
