```python
import gluoncv.model_zoo as gz
from gluoncv.utils import export_block
def zoo_import(name):
	"""Download from Gluoncv Zoo"""
	net = gz.get_model(name, pretrained=True)
	export_block(name, net, preprocess=True)
	print(net)
zoo_import('yolo3_darknet53_coco')
```
