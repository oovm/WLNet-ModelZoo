(* ::Package:: *)

SetDirectory@NotebookDirectory[];
Needs["MXNetLink`"]
Needs["NeuralNetworks`"]
DateString[]


(* ::Subitem:: *)
(*Tue 16 Oct 2018 15:56:56*)


(* ::Subchapter:: *)
(*Import Weights*)


ndarry = NDArrayImport["cifar_resnet56_v2-0000.params"];


(* ::Subchapter:: *)
(*Encoder & Decoder*)


mShift = {0.485, 0.456, 0.406};
vShift = {0.229, 0.224, 0.225}^2;
encoder = NetEncoder[{"Image", 32, "MeanImage" -> mShift, "VarianceImage" -> vShift}]
tags = {"airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"};
decoder = NetDecoder[{"Class", tags}]


(* ::Subchapter:: *)
(*Pre-defined Structure*)


getBN[i_, j_] := BatchNormalizationLayer[
	"Epsilon" -> 1*^-5,
	"Beta" -> ndarry["arg:cifarresnetv21_stage" <> i <> "_batchnorm" <> j <> "_beta"],
	"Gamma" -> ndarry["arg:cifarresnetv21_stage" <> i <> "_batchnorm" <> j <> "_gamma"],
	"MovingMean" -> ndarry["aux:cifarresnetv21_stage" <> i <> "_batchnorm" <> j <> "_running_mean"],
	"MovingVariance" -> ndarry["aux:cifarresnetv21_stage" <> i <> "_batchnorm" <> j <> "_running_var"]
]
getBN2[j_] := BatchNormalizationLayer[
	"Epsilon" -> 1*^-5,
	"Beta" -> ndarry["arg:cifarresnetv21_batchnorm" <> j <> "_beta"],
	"Gamma" -> ndarry["arg:cifarresnetv21_batchnorm" <> j <> "_gamma"],
	"MovingMean" -> ndarry["aux:cifarresnetv21_batchnorm" <> j <> "_running_mean"],
	"MovingVariance" -> ndarry["aux:cifarresnetv21_batchnorm" <> j <> "_running_var"]
]
getCN[i_, j_, s_ : 1, p_ : 1] := ConvolutionLayer[
	"Weights" -> ndarry["arg:cifarresnetv21_stage" <> i <> "_conv" <> j <> "_weight"],
	"Biases" -> None, "PaddingSize" -> p, "Stride" -> s
]
getCN2[j_] := ConvolutionLayer[
	"Weights" -> ndarry["arg:cifarresnetv21_conv" <> j <> "_weight"],
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
	AggregationLayer[Mean],
	LinearLayer[10,
		"Weights" -> ndarry["arg:cifarresnetv21_dense0_weight"],
		"Biases" -> ndarry["arg:cifarresnetv21_dense0_bias"]
	]
}]


(* ::Subchapter:: *)
(*Main*)


mainNet = NetChain[{
	NetChain[{getBN2["0"], getCN2["0"], getBlock[1, 0, 0]}],
	NetChain@Table[getBlock[1, i, 0], {i, 2, 16, 2}],
	getBlock2[2, 0],
	NetChain@Table[getBlock[2, i, 1], {i, 2, 16, 2}],
	getBlock2[3, 0],
	NetChain@Table[getBlock[3, i, 1], {i, 2, 16, 2}],
	getBlock3[],
	SoftmaxLayer[]
},
	"Input" -> encoder,
	"Output" -> decoder
]


(* ::Subchapter:: *)
(*Export Model*)


Export["cifar_resnet56_v2.WMLF", mainNet]
