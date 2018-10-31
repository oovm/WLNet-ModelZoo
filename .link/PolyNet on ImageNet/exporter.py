#!/usr/bin/env python
# coding=utf-8
import re

import torch
import wolframclient.serializers as wxf
from pretrainedmodels import polynet

# manually fix this first
model = polynet(num_classes=1000, pretrained='imagenet')
net = list(model.modules())
params = model.state_dict()

# remove `bn.num_batches_tracked` because it can broke the model
npy = {
    key: value.numpy()
    for key, value
    in params.items()
    if not re.match('.*_tracked$', key)
}

wxf.export(npy, 'imagenet_polynet.wxf', target_format='wxf')
torch.save(polynet(num_classes=1000), 'imagenet_polynet.pth')
