#!/usr/bin/env python
# coding=utf-8
import numpy as npy
from numpy import random
import wolframclient.serializers as wxf


def npy2wxf(path):
	data = npy.load(path)
	wxf.export(data, path + '.wxf', target_format='wxf')


npy.save('4d_array.npy', random.rand(1, 3, 32, 32))
npy2wxf('4d_array.npy')