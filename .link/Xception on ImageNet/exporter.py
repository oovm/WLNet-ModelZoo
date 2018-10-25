from os.path import abspath
from sys import path

path.append(abspath('../.tools/'))
import torch
from pth_exporter import pth2wxf
from pretrainedmodels import xception

# manually fix this first
net = xception(num_classes=1000, pretrained=False)
torch.save(net, 'imagenet_xception.pth')

