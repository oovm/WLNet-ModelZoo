```python
import gluoncv.model_zoo as gz
from gluoncv.utils import export_block
def zoo_import(name):
	"""Download from Gluoncv Zoo"""
	net = gz.get_model(name, pretrained=True)
	export_block(name, net, preprocess=True)
	print(net)

zoo_import('cifar_resnet20_v2')
zoo_import('cifar_resnet56_v2')
zoo_import('cifar_resnet110_v2')

zoo_import('cifar_wideresnet16_10')
zoo_import('cifar_wideresnet28_10')
zoo_import('cifar_wideresnet40_8')
```


