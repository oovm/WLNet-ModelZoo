



# Resnet trained on ImageNet

 3\*224\*224 Image

```Mathematica
bn = BatchNormalizationLayer["Epsilon" -> 1*^-5];
relu = ElementwiseLayer["ReLU"];
head = {
	bn, ConvolutionLayer[64, {7, 7}, "PaddingSize" -> 3, "Stride" -> 2],
	relu, PoolingLayer[{3, 3}, "Function" -> Mean, "Stride" -> 2]
};
tail = {
	bn, relu,
	PoolingLayer[{7, 7}, "PaddingSize" -> 0, "Stride" -> 7],
	FlattenLayer[]
};
predict = {1000, SoftmaxLayer[]};
NetChain[{
	head,
	ResBlockV2[64, 2, False],
	ResBlockV2[128, 4, True],
	ResBlockV2[256, 8, True],
	tail, predict
}]
```