import re
import torch
import wolframclient.serializers as wxf


def pth2wxf(path):
    pth = torch.load(path, map_location=torch.device('cpu'))
    npy = {
        key: value.numpy()
        for key, value
        in pth.items()
        if not re.match('.*_tracked$', key)
    }
    wxf.export(npy, path + '.wxf', target_format='wxf')


pth2wxf('resnet18.pth')
pth2wxf('resnet34.pth')
pth2wxf('resnet50.pth')
