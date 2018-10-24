import gluoncv.model_zoo as gz
from gluoncv.utils import export_block


def zoo_import(name, head=''):
    """Download from Gluoncv Zoo"""
    net = gz.get_model(name, pretrained=True)
    export_block(head + name, net, preprocess=True)


zoo_import('vgg11', 'imagenet_')
zoo_import('vgg13', 'imagenet_')
zoo_import('vgg16', 'imagenet_')
zoo_import('vgg19', 'imagenet_')
