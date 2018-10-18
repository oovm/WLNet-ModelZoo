(* ::Package:: *)

SetDirectory@NotebookDirectory[];
Needs["MXNetLink`"]
Needs["NeuralNetworks`"]
DateString[]


(* ::Subitem:: *)
(*Thu 18 Oct 2018 19:08:29*)


(* ::Subchapter:: *)
(*Import Weights*)


ndarry = NDArrayImport["imagenet_alexnet-0000.params"];


(* ::Subchapter:: *)
(*Encoder & Decoder*)


mShift = {0.485, 0.456, 0.406};
vShift = {0.229, 0.224, 0.225}^2;
encoder = NetEncoder[{"Image", 224, "MeanImage" -> mShift, "VarianceImage" -> vShift}]
decoder = NetExtract[NetModel["ResNet-50 Trained on ImageNet Competition Data"], "Output"]


(* ::Subchapter:: *)
(*Pre-defined Structure*)


getCN[i_, p_, s_] := ConvolutionLayer[
	"Weights" -> ndarry["arg:alexnet0_conv" <> i <> "_weight"],
	"Biases" -> ndarry["arg:alexnet0_conv" <> i <> "_bias"],
	"PaddingSize" -> p, "Stride" -> s
]
getFC[i_, n_] := LinearLayer[n,
	"Weights" -> ndarry["arg:alexnet0_dense" <> i <> "_weight"],
	"Biases" -> ndarry["arg:alexnet0_dense" <> i <> "_bias"]
]


(* ::Subchapter:: *)
(*Main*)


mainNet = NetChain[{
	getCN["0", 2, 4], Ramp,
	PoolingLayer[{3, 3}, "Stride" -> 2],
	getCN["1", 2, 1], Ramp,
	PoolingLayer[{3, 3}, "Stride" -> 2],
	getCN["2", 1, 1], Ramp,
	getCN["3", 1, 1], Ramp,
	getCN["4", 1, 1], Ramp,
	PoolingLayer[{3, 3}, "Stride" -> 2],
	FlattenLayer[],
	getFC["0", 4096], Ramp,
	DropoutLayer[0.5],
	getFC["1", 4096], Ramp,
	DropoutLayer[0.5],
	getFC["2", 1000],
	SoftmaxLayer[]
},
	"Input" -> encoder ,
	"Output" -> decoder
]


(* ::Subchapter:: *)
(*Export Model*)


Export["imagenet_alexnet.WMLF", mainNet]
