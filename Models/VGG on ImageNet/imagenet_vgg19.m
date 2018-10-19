(* ::Package:: *)

SetDirectory@NotebookDirectory[];
Needs["MXNetLink`"]
Needs["NeuralNetworks`"]
DateString[]


(* ::Subitem:: *)
(*Fri 19 Oct 2018 16:23:22*)


(* ::Subchapter:: *)
(*Import Weights*)


ndarry = NDArrayImport["imagenet_vgg19-0000.params"];


(* ::Subchapter:: *)
(*Encoder & Decoder*)


mShift = {0.485, 0.456, 0.406};
vShift = {0.229, 0.224, 0.225}^2;
encoder = NetEncoder[{"Image", 224, "MeanImage" -> mShift, "VarianceImage" -> vShift}]
decoder = NetExtract[NetModel["ResNet-50 Trained on ImageNet Competition Data"], "Output"]


(* ::Subchapter:: *)
(*Pre-defined Structure*)


getCN[i_] := ConvolutionLayer[
	"Weights" -> ndarry["arg:vgg3_conv" <> i <> "_weight"],
	"Biases" -> ndarry["arg:vgg3_conv" <> i <> "_bias"],
	"PaddingSize" -> 1, "Stride" -> 1
]
getFC[i_, n_] := LinearLayer[n,
	"Weights" -> ndarry["arg:vgg3_dense" <> i <> "_weight"],
	"Biases" -> ndarry["arg:vgg3_dense" <> i <> "_bias"]
]
getBlock[i_, j_] := NetChain@Flatten@Table[{getCN[ToString@n], Ramp}, {n, i, j}]
getBlock2[i_, j_] := NetChain@{getFC[ToString@i, j], Ramp, DropoutLayer[0.5]}
pool = PoolingLayer[{2, 2}, "Stride" -> 2]


(* ::Subchapter:: *)
(*Main*)


mainNet = NetChain[{
	getBlock[0, 1], pool,
	getBlock[2, 3], pool,
	getBlock[4, 7], pool,
	getBlock[8, 11], pool,
	getBlock[12, 15], pool,
	getBlock2[0, 4096],
	getBlock2[1, 4096],
	getFC["2", 1000]
},
	"Input" -> encoder, "Output" -> decoder
]


(* ::Subchapter:: *)
(*Export Model*)


Export["imagenet_vgg19.WMLF", mainNet]
