# !/usr/bin/env python
# -*- coding:utf-8 -*-
""" 
@author: Wang Ye (Wayne)
@file: main.py
@time: 2020/10/21
@contact: wangye@oppo.com
@site: 
@software: PyCharm
# code is far away from bugs.
"""

import pretrainedmodels
from IPython.display import display
import pprint
import pretrainedmodels.utils as utils

# [print(name) for name in pretrainedmodels.model_names]
# pprint.pprint(pretrainedmodels.pretrained_settings['nasnetalarge'])
model_name = 'nasnetalarge'
model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
load_img = utils.LoadImage()
tf_img = utils.TransformImage(model)
