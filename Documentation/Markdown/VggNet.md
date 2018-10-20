





```Mathematica
Vgg19 := NetChain@{
	VggBlock[64, 2, "BN"],
	VggBlock[128, 2, "BN"],
	VggBlock[256, 4, "BN"],
	VggBlock[512, 4, "BN"],
	VggBlock[512, 4, "BN"],
	{4096, Ramp, DropoutLayer[0.5]},
	{4096, Ramp, DropoutLayer[0.5]},
	1000
};
```