import torch

import wolframclient.serializers as wxf

'''
import torch.onnx as onnx
form .python.models import * 
model = Net4x().cuda()
model.load_state_dict(torch.load('./model/a4/model_new.pth')).cuda()
dummy_input = Variable(torch.randn(1, 1, 36, 36)).cuda()
onnx.export(model, dummy_input, "a4.onnx", verbose=True)
'''


def exportSR(name):
    pth = torch.load('./model/' + name + '/model_new.pth')
    npy = {key: value.numpy() for key, value in pth.items()}
    wxf.export(npy, 'moe_' + name + '.wxf', target_format='wxf')


def exportDN(name):
    pth = torch.load('./model/' + name + '/model_new.pth')
    npy = {key: value.numpy() for key, value in pth.items()}
    wxf.export(npy, 'moe_' + name + '.wxf', target_format='wxf')


def exportDH(name):
    pth = torch.load('./model/' + name + '/AOD_net_epoch_relu_10.pth')
    npy = {key: value.numpy() for key, value in pth.items()}
    wxf.export(npy, 'moe_' + name + '.wxf', target_format='wxf')


[exportSR(i) for i in ['a2', 'a3', 'a4', 'p2', 'p3', 'p4']]
[exportDN(i) for i in ['l15', 'l25', 'l50', 'dn_lite5', 'dn_lite10', 'dn_lite15']]
[exportDH(i) for i in ['dehaze']]
