(* ::Package:: *)

SetDirectory@NotebookDirectory[];
Needs["MXNetLink`"]
Needs["NeuralNetworks`"]
DateString[]


(* ::Subitem:: *)
(*Wed 17 Oct 2018 12:38:48*)


(* ::Subchapter:: *)
(*Import Weights*)


ndarry = NDArrayImport["imagenet_densenet169-0000.params"];


(* ::Subchapter:: *)
(*Encoder & Decoder*)


mShift = {0.485, 0.456, 0.406};
vShift = {0.229, 0.224, 0.225}^2;
encoder = NetEncoder[{"Image", 224, "MeanImage" -> mShift, "VarianceImage" -> vShift}]
decoder = NetExtract[NetModel["ResNet-50 Trained on ImageNet Competition Data"], "Output"]


(* ::Subchapter:: *)
(*Pre-defined Structure*)


getBN[i_, j_] := BatchNormalizationLayer[
	"Epsilon" -> 1*^-5,
	"Beta" -> ndarry["arg:densenet2_stage" <> i <> "_batchnorm" <> j <> "_beta"],
	"Gamma" -> ndarry["arg:densenet2_stage" <> i <> "_batchnorm" <> j <> "_gamma"],
	"MovingMean" -> ndarry["aux:densenet2_stage" <> i <> "_batchnorm" <> j <> "_running_mean"],
	"MovingVariance" -> ndarry["aux:densenet2_stage" <> i <> "_batchnorm" <> j <> "_running_var"]
]
getBN2[j_] := BatchNormalizationLayer[
	"Epsilon" -> 1*^-5,
	"Beta" -> ndarry["arg:densenet2_batchnorm" <> j <> "_beta"],
	"Gamma" -> ndarry["arg:densenet2_batchnorm" <> j <> "_gamma"],
	"MovingMean" -> ndarry["aux:densenet2_batchnorm" <> j <> "_running_mean"],
	"MovingVariance" -> ndarry["aux:densenet2_batchnorm" <> j <> "_running_var"]
]
getCN[i_, j_, p_ : 1, s_ : 1] := ConvolutionLayer[
	"Weights" -> ndarry["arg:densenet2_stage" <> i <> "_conv" <> j <> "_weight"],
	"Biases" -> None, "PaddingSize" -> p, "Stride" -> s
]
getCN2[j_, p_ : 1, s_ : 1] := ConvolutionLayer[
	"Weights" -> ndarry["arg:densenet2_conv" <> j <> "_weight"],
	"Biases" -> None, "PaddingSize" -> p, "Stride" -> s
]
$getBlock = NetChain@{
	getCN2["0", 3, 2], getBN2["0"], Ramp,
	PoolingLayer[{3, 3}, "Stride" -> 2, "PaddingSize" -> 1]
}
getBlock[i_, j_] := NetGraph[{
	getBN[ToString[i], ToString[j]], Ramp, getCN[ToString[i], ToString[j], 0, 1],
	getBN[ToString[i], ToString[j + 1]], Ramp, getCN[ToString[i], ToString[j + 1]],
	CatenateLayer[]
}, {
	NetPort["Input"] -> 7,
	NetPort["Input"] -> 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7
}]
getBlock2[i_] := NetChain@{
	getBN2[ToString@i], Ramp, getCN2[ToString@i, 0, 1],
	PoolingLayer[{2, 2}, "Stride" -> 2, "Function" -> Mean]
}
$getBlock2 = NetChain@{
	getBN2["4"], Ramp,
	PoolingLayer[{7, 7}, "Stride" -> 7, "Function" -> Mean]
	
}


(* ::Subchapter:: *)
(*Main*)


mainNet = NetChain[{
	$getBlock,
	NetChain@Table[getBlock[1, i], {i, 0, 10, 2}],
	getBlock2[1],
	NetChain@Table[getBlock[2, i], {i, 0, 22, 2}],
	getBlock2[2],
	NetChain@Table[getBlock[3, i], {i, 0, 62, 2}],
	getBlock2[3],
	NetChain@Table[getBlock[4, i], {i, 0, 62, 2}],
	$getBlock2,
	LinearLayer[1000,
		"Weights" -> ndarry["arg:densenet2_dense0_weight"],
		"Biases" -> ndarry["arg:densenet2_dense0_bias"]
	],
	SoftmaxLayer[]
},
	"Input" -> encoder,
	"Output" -> decoder
]


(* ::Subchapter:: *)
(*Export Model*)


Export["imagenet_densenet169.WMLF", mainNet]
