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
import requests
import torch

# [print(name) for name in pretrainedmodels.model_names]
# pprint.pprint(pretrainedmodels.pretrained_settings['nasnetalarge'])
model_name = 'nasnetalarge'
model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
model.eval()
load_img = utils.LoadImage()
tf_img = utils.TransformImage(model)
# cat = r'https://raw.githubusercontent.com/Cadene/pretrained-models.pytorch/master/data/cat.jpg'
# res = requests.get(cat).content
# import os
# os.makedirs('data')
# with open('data/cat.jpg','wb') as f:
#     f.write(res)

path_img = 'data/cat.jpg'

input_img = load_img(path_img)
input_tensor = tf_img(input_img)  # 3x400x225 -> 3x299x299 size may differ
input_tensor = input_tensor.unsqueeze(0)  # 3x299x299 -> 1x3x299x299
input = torch.autograd.Variable(input_tensor, requires_grad=False)

output_logits = model(input)  # 1x1000
print(output_logits)
# output_features = model.features(input) # 1x14x14x2048 size may differ
# output_logits = model.logits(output_features) # 1x1000
