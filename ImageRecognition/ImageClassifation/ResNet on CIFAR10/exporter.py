import gluoncv.model_zoo as gz
from gluoncv.utils import export_block


def zoo_import(name, head=''):
    """Download from Gluoncv Zoo"""
    net = gz.get_model(name, pretrained=True)
    export_block(head + name, net, preprocess=True)


zoo_import('cifar_resnet20_v2')
zoo_import('cifar_resnet56_v2')
zoo_import('cifar_resnet110_v2')
