#!/usr/bin/env python
# coding=utf-8
import torch
import wolframclient.serializers as wxf


def pth2wxf(path):
	pth = torch.load(path, map_location=torch.device('cpu'))
	npy = {key: value.numpy() for key, value in pth.items()}
	wxf.export(npy, path + '.wxf', target_format='wxf')


pth2wxf('checkpoint.pth')
