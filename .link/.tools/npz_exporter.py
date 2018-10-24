#!/usr/bin/env python
# coding=utf-8
import numpy as npz
from numpy import random
import wolframclient.serializers as wxf


def npz2wxf(path):
	data = npz.load(path)
	wxf.export(data, path + '.wxf', target_format='wxf')


a = npz.array([[1, 2, 3], [4, 5, 6]])
b = npz.arange(0, 1.0, 0.1)
c = npz.sin(b)
npz.savez("result.npz", a, b, sin_arr=c)
r = npz.load("result.npz")
print(r.all())