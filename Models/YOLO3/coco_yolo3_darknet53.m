(* ::Package:: *)

SetDirectory@NotebookDirectory[]; Now
<< MXNetLink`
<< NeuralNetworks`


file = "yolo3_darknet53_coco";
ndarry = NDArrayImport[file <> "-0000.params"];
params = MXModelLoadParameters[file <> "-0000.params"];


input = NetEncoder[{"Image", 224, "MeanImage" -> {.485, .456, .406}, "VarianceImage" -> {.229, .224, .225}^2}]
leayReLU[alpha_] := ElementwiseLayer[Ramp[#] - alpha * Ramp[-#]&]
getCV$a[n_, p_ : 1, s_ : 1] := ConvolutionLayer[
	"Weights" -> ndarry["arg:darknetv30_conv" <> ToString[n] <> "_weight"],
	"Biases" -> None, "PaddingSize" -> p, "Stride" -> s
];
getBN$a[n_] := BatchNormalizationLayer[
	"Epsilon" -> 1*^-5,
	"Beta" -> ndarry["arg:darknetv30_batchnorm" <> ToString[n] <> "_beta"],
	"Gamma" -> ndarry["arg:darknetv30_batchnorm" <> ToString[n] <> "_gamma"],
	"MovingMean" -> ndarry["aux:darknetv30_batchnorm" <> ToString[n] <> "_running_mean"],
	"MovingVariance" -> ndarry["aux:darknetv30_batchnorm" <> ToString[n] <> "_running_var"]
];
getCV$b[n_, p_ : 1, s_ : 1] := ConvolutionLayer[
	"Weights" -> ndarry["arg:yolov30_yolodetectionblockv30_conv" <> ToString[n] <> "_weight"],
	"Biases" -> None, "PaddingSize" -> p, "Stride" -> s
];
getBN$b[n_] := BatchNormalizationLayer[
	"Epsilon" -> 1*^-5,
	"Beta" -> ndarry["arg:yolov30_yolodetectionblockv30_batchnorm" <> ToString[n] <> "_beta"],
	"Gamma" -> ndarry["arg:yolov30_yolodetectionblockv30_batchnorm" <> ToString[n] <> "_gamma"],
	"MovingMean" -> ndarry["aux:yolov30_yolodetectionblockv30_batchnorm" <> ToString[n] <> "_running_mean"],
	"MovingVariance" -> ndarry["aux:yolov30_yolodetectionblockv30_batchnorm" <> ToString[n] <> "_running_var"]
];


block1 = NetChain[{
	getCV$a[0, 1, 1], getBN$a[0], leayReLU[0.1],
	getCV$a[1, 1, 2], getBN$a[1], leayReLU[0.1]
}]


getBlock$a[n_] := NetGraph[{
	NetChain[{
		getCV$a[n, 0, 1], getBN$a[n], leayReLU[0.1],
		getCV$a[n + 1, 1, 1], getBN$a[n + 1], leayReLU[0.1]
	}],
	ThreadingLayer[Plus]
}, {
	NetPort["Input"] -> 1 -> 2,
	NetPort["Input"] -> 2
}]
getBlock$b[n_] := NetChain[{getCV$a[n, 1, 2], getBN$a[n], leayReLU[0.1]}]
block2 = NetChain@Flatten[{
	{getBlock$a[2], getBlock$b[4]},
	{getBlock$a[5], getBlock$a[7], getBlock$b[9]},
	getBlock$a /@ Range[10, 24, 2]
}]
block3 = NetChain@Join[{getBlock$b[26]}, getBlock$a /@ Range[27, 41, 2]]


block4 = NetChain@Join[{getBlock$b[43]}, getBlock$a /@ Range[44, 50, 2]]
block5 = NetChain[{
	getCV$b[0, 0, 1], getBN$b[0], leayReLU[0.1],
	getCV$b[1, 1, 1], getBN$b[1], leayReLU[0.1],
	getCV$b[2, 0, 1], getBN$b[2], leayReLU[0.1],
	getCV$b[3, 1, 1], getBN$b[3], leayReLU[0.1],
	getCV$b[4, 0, 1], getBN$b[4], leayReLU[0.1]
}]


NetChain[{getCV$b[5, 0, 1], getBN$b[5], leayReLU[0.1]}]


Take[Keys@ndarry, {263, -1}] // TableForm


NetModel[]


NetModel["YOLO V2 Trained on MS-COCO Data", "ConstructionNotebook"]


regionLayerNet[{w, h}, anchors, coord, classes]
