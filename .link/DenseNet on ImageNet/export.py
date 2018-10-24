import gluoncv.model_zoo as gz
from gluoncv.utils import export_block

def zoo_import(name, head=''):
	"""Download from Gluoncv Zoo"""
	net = gz.get_model(name, pretrained=True)
	export_block(head + name, net, preprocess=True)

zoo_import('densenet121', 'imagenet_')
zoo_import('densenet161', 'imagenet_')
zoo_import('densenet169', 'imagenet_')
zoo_import('densenet201', 'imagenet_')