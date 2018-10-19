import gluoncv.model_zoo as gz
from gluoncv.utils import export_block

def zoo_import(name, head=''):
	"""Download from Gluoncv Zoo"""
	net = gz.get_model(name, pretrained=True)
	export_block(head + name, net, preprocess=True)

# can not download
# zoo_import('fcn_resnet50_voc')
# zoo_import('fcn_resnet101_voc')