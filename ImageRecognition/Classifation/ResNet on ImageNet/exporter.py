import gluoncv.model_zoo as gz
from gluoncv.utils import export_block


def zoo_import(name, head=''):
    """Download from Gluoncv Zoo"""
    net = gz.get_model(name, pretrained=True)
    export_block(head + name, net, preprocess=True)


zoo_import('resnet18_v1b', "imagenet_")
zoo_import('resnet34_v1b', "imagenet_")
zoo_import('resnet50_v1s', "imagenet_")
zoo_import('resnet101_v1s', "imagenet_")
zoo_import('resnet152_v1s', "imagenet_")
