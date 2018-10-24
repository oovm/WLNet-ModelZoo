import gluoncv.model_zoo as gz
from gluoncv.utils import export_block

def zoo_import(name, head=''):
	"""Download from Gluoncv Zoo"""
	net = gz.get_model(name, pretrained=True)
	export_block(head + name, net, preprocess=True)

# can not download
# zoo_import('resnext50_32x4d')
# zoo_import('resnext101_32x4d')
# zoo_import('resnext101_64x4d')
