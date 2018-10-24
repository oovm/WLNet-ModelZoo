import gluoncv.model_zoo as gz
from gluoncv.utils import export_block


def zoo_import(name, head=''):
    """Download from Gluoncv Zoo"""
    net = gz.get_model(name, pretrained=True)
    export_block(head + name, net, preprocess=True)


# can not download
# zoo_import('cifar_resnext29_32x4d')
zoo_import('cifar_resnext29_16x64d')
