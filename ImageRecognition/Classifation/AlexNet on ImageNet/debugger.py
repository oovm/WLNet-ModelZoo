from argparse import Namespace

import wolframclient.serializers as wxf
from gluoncv.data.transforms.presets.imagenet import transform_eval
from gluoncv.model_zoo import get_model
from mxnet import image, symbol, gluon

params = Namespace(
    model='alexnet',
    input_pic='./ILSVRC2012_val_00000001.png',
    debug_nodes=[
        "alexnet0_conv0_fwd_output",
        "alexnet0_conv1_fwd_output",
        "alexnet0_conv2_fwd_output",
        "alexnet0_conv3_fwd_output",
        "alexnet0_conv4_fwd_output",
        "alexnet0_dense0_fwd_output",
        "alexnet0_dense1_fwd_output",
        "alexnet0_dense2_fwd_output",
        "alexnet0_dropout0_fwd_output",
        "alexnet0_dropout1_fwd_output",
        "alexnet0_flatten0_flatten0_output",
        "alexnet0_pool0_fwd_output",
        "alexnet0_pool1_fwd_output",
        "alexnet0_pool2_fwd_output"
    ]
)

# 预处理模型以及图片
net = get_model(params.model, pretrained=True)
img = image.imread(params.input_pic)
img = transform_eval(img, resize_short=224, crop_size=224)
wxf.export(img.asnumpy(), 'input.wxf', target_format='wxf')

# 列出可选的输出节点
nodes = net(symbol.var('flow')).get_internals().list_outputs()
wxf.export(nodes, 'nodes.wxf', target_format='wxf')


def debug_net(net):
    data = symbol.var('flow')
    internals = net(data).get_internals()
    hooks = [internals[i] for i in params.debug_nodes]
    new = gluon.SymbolBlock(hooks, data, params=net.collect_params())
    return new


debug = debug_net(net)
ndarray = [i.asnumpy() for i in debug(img)]
wxf.export(ndarray, 'debug.wxf', target_format='wxf')
