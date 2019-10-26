import gzip
import shutil

import caffe2.proto.caffe2_pb2 as caffe
import numpy as npy
import wolframclient.serializers as wxf


def npy2wxf(path):
    data = npy.load(path)
    wxf.export(data, path + '.wxf', target_format='wxf')


def caffe2wxf(path):
    data = dict()
    model = caffe.NetParameter()
    model.ParseFromString(open(path, 'rb').read())

    get_array = lambda p: npy.array(p.data, dtype='float32').reshape(p.shape.dim)

    def add_array(node):
        for i in range(len(node.blobs)):
            data[node.name + "_" + str(i + 1)] = get_array(node.blobs[i])

    def add_node(layers):
        for i in range(len(layers)):
            add_array(layers[i])

    add_node(model.layer)
    wxf.export(data, path + '.wxf', target_format='wxf')


shutil.copyfileobj(gzip.open('tag_list.json.gz', 'rb'), open('tag_list.json', 'wb'))
npy2wxf("image_mean.npy")
caffe2wxf("illust2vec_ver200.caffemodel")
caffe2wxf("illust2vec_tag_ver200.caffemodel")
