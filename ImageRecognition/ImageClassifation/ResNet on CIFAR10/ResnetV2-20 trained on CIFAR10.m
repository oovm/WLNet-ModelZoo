(* ::Package:: *)

SetDirectory@NotebookDirectory[];
Needs["MXNetLink`"]
Needs["NeuralNetworks`"]
DateString[]


(* ::Subitem:: *)
(*Fri 16 Nov 2018 14:07:00*)


(* ::Subchapter:: *)
(*Import Weights*)


params = NDArrayImport["cifar_resnet20_v2-0000.params"];


(* ::Subchapter:: *)
(*Encoder & Decoder*)


mShift = {0.485, 0.456, 0.406};
vShift = {0.229, 0.224, 0.225}^2;
encoder = NetEncoder[{"Image", 32, "MeanImage" -> mShift, "VarianceImage" -> vShift}]
tags = {"airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"};
decoder = NetDecoder[{"Class", tags}]


(* ::Subchapter::Closed:: *)
(*Pre-defined Structure*)


getBN[i_, j_] := BatchNormalizationLayer[
	"Epsilon" -> 1*^-5,
	"Beta" -> params["arg:cifarresnetv20_stage" <> i <> "_batchnorm" <> j <> "_beta"],
	"Gamma" -> params["arg:cifarresnetv20_stage" <> i <> "_batchnorm" <> j <> "_gamma"],
	"MovingMean" -> params["aux:cifarresnetv20_stage" <> i <> "_batchnorm" <> j <> "_running_mean"],
	"MovingVariance" -> params["aux:cifarresnetv20_stage" <> i <> "_batchnorm" <> j <> "_running_var"]
]
getBN2[j_] := BatchNormalizationLayer[
	"Epsilon" -> 1*^-5,
	"Beta" -> params["arg:cifarresnetv20_batchnorm" <> j <> "_beta"],
	"Gamma" -> params["arg:cifarresnetv20_batchnorm" <> j <> "_gamma"],
	"MovingMean" -> params["aux:cifarresnetv20_batchnorm" <> j <> "_running_mean"],
	"MovingVariance" -> params["aux:cifarresnetv20_batchnorm" <> j <> "_running_var"]
]
getCN[i_, j_, s_ : 1, p_ : 1] := ConvolutionLayer[
	"Weights" -> params["arg:cifarresnetv20_stage" <> i <> "_conv" <> j <> "_weight"],
	"Biases" -> None, "PaddingSize" -> p, "Stride" -> s
]
getCN2[j_] := ConvolutionLayer[
	"Weights" -> params["arg:cifarresnetv20_conv" <> j <> "_weight"],
	"Biases" -> None, "PaddingSize" -> 1
]
getBlock[i_, j_, k_ : 0] := NetGraph[{
	getBN[ToString[i], ToString[j]], Ramp, getCN[ToString[i], ToString[j + k]],
	getBN[ToString[i], ToString[j + 1]], Ramp, getCN[ToString[i], ToString[j + 1 + k]],
	ThreadingLayer[Plus]
}, {
	NetPort["Input"] -> 7,
	NetPort["Input"] -> 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7
}]
getBlock2[i_, j_] := NetGraph[{
	getBN[ToString[i], ToString[j + 0]], Ramp, getCN[ToString[i], ToString[j + 0], 2, 1],
	getBN[ToString[i], ToString[j + 1]], Ramp, getCN[ToString[i], ToString[j + 1], 1, 1],
	getCN[ToString[i], ToString[j + 2], 2, 0], ThreadingLayer[Plus]
}, {
	NetPort["Input"] -> 7 -> 8,
	NetPort["Input"] -> 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 8
}]
getBlock3[] := NetChain[{
	getBN2["1"], Ramp,
	AggregationLayer[Mean]
}]


(* ::Subchapter:: *)
(*Main*)


extractor = NetChain[{
	getBN2["0"],
	getCN2["0"],
	getBlock[1, 0, 0],
	NetChain@Table[getBlock[1, i, 0], {i, 2, 4, 2}],
	getBlock2[2, 0],
	NetChain@Table[getBlock[2, i, 1], {i, 2, 4, 2}],
	getBlock2[3, 0],
	NetChain@Table[getBlock[3, i, 1], {i, 2, 4, 2}],
	getBlock3[]
}];
classifier = LinearLayer[10,
	"Weights" -> params["arg:cifarresnetv20_dense0_weight"],
	"Biases" -> params["arg:cifarresnetv20_dense0_bias"]
];
mainNet = NetChain[{
	"Extractor" -> extractor,
	"Classifier" -> classifier,
	"Predictor" -> SoftmaxLayer[]
},
	"Input" -> encoder,
	"Output" -> decoder
]


(* ::Subchapter:: *)
(*Export Model*)


Export["ResnetV2-20 trained on CIFAR10.WXF", mainNet]
