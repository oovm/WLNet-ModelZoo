import gluoncv.model_zoo as gz
from gluoncv.utils import export_block

def zoo_import(name, head=''):
	"""Download from Gluoncv Zoo"""
	net = gz.get_model(name, pretrained=True)
	export_block(head + name, net, preprocess=True)

zoo_import('vgg11_bn', 'imagenet_')
zoo_import('vgg13_bn', 'imagenet_')
zoo_import('vgg16_bn', 'imagenet_')
zoo_import('vgg19_bn', 'imagenet_')