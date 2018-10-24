#!/usr/bin/env python
# coding=utf-8
import pickle as pkl
import wolframclient.serializers as wxf


def pkl2wxf(path):
	file = open(path, 'rb')
	objs = []
	while True:
		try : objs.append(pkl.load(file))
		except EOFError : break
	file.close()
	print(objs)
	wxf.export(objs, path + '.wxf', target_format='wxf')


f = open('objs.pkl', 'wb')
# Test basic types
testDict = {
	0: None,
	1: [1, 2, 3, 4],
	2: ('true', 'false'),
	3: {'yes': True, 'no': False}
}
pkl.dump(testDict, f)
f.close()
pkl2wxf('objs.pkl')