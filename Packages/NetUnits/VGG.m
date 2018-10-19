(* ::Package:: *)
(* ::Subsection::Closed:: *)
(*附加设置*)
VggBlockBN::usage = "这里应该填这个函数的说明,如果要换行用\"\\r\"\r就像这样";
VggBlockOriginal::usage = "这里应该填这个函数的说明,如果要换行用\"\\r\"\r就像这样";
(* ::Subsection::Closed:: *)
(*附加设置*)
Begin["`VGG`"];
VggBlockOriginal[c_Integer, u_Integer] := Block[
	{unit, pool},
	unit = {
		ConvolutionLayer[c, {3, 3}, "PaddingSize" -> 1, "Stride" -> 1],
		ElementwiseLayer["ReLU"]
	};
	pool = PoolingLayer[{2, 2}, "Stride" -> 2];
	NetChain@Flatten[{ConstantArray[unit, u], pool}]
];
VggBlockBN[c_Integer, u_Integer] := Block[
	{unit, pool},
	unit = {
		ConvolutionLayer[c, {3, 3}, "PaddingSize" -> 1, "Stride" -> 1],
		BatchNormalizationLayer["Epsilon" -> 1*^-5],
		ElementwiseLayer["ReLU"]
	};
	pool = PoolingLayer[{2, 2}, "Stride" -> 2];
	NetChain@Flatten[{ConstantArray[unit, u], pool}]
];

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
(* ::Subsection:: *)
(*Additional*)
SetAttributes[
	{ },
	{Protected, ReadProtected}
];
End[]