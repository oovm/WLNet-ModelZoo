(* ::Package:: *)

SetDirectory@NotebookDirectory[];
Needs["MXNetLink`"]
Needs["NeuralNetworks`"]
DateString[]


(* ::Subitem:: *)
(*Fri 19 Oct 2018 16:29:40*)


(* ::Subchapter:: *)
(*Import Weights*)


params = NDArrayImport["imagenet_vgg11-0000.params"];


(* ::Subchapter:: *)
(*Encoder & Decoder*)


mShift = {0.485, 0.456, 0.406};
vShift = {0.229, 0.224, 0.225}^2;
encoder = NetEncoder[{"Image", 224, "MeanImage" -> mShift, "VarianceImage" -> vShift}]
decoder = NetExtract[NetModel["ResNet-50 Trained on ImageNet Competition Data"], "Output"]


(* ::Subchapter:: *)
(*Pre-defined Structure*)


getCN[i_] := ConvolutionLayer[
	"Weights" -> params["arg:vgg0_conv" <> i <> "_weight"],
	"Biases" -> params["arg:vgg0_conv" <> i <> "_bias"],
	"PaddingSize" -> 1, "Stride" -> 1
]
getFC[i_, n_] := LinearLayer[n,
	"Weights" -> params["arg:vgg0_dense" <> i <> "_weight"],
	"Biases" -> params["arg:vgg0_dense" <> i <> "_bias"]
]
pool := PoolingLayer[{2, 2}, "Stride" -> 2]
getBlock[i_, j_] := NetChain@Flatten[{
Table[{getCN[ToString@n], Ramp}, {n, i, j}],
pool
}]

getBlock2[i_, j_] := NetChain@{getFC[ToString@i, j], Ramp, DropoutLayer[0.5]}



(* ::Subchapter:: *)
(*Main*)


mainNet = NetChain[{
	getBlock[0, 0], 
	getBlock[1, 1], 
	getBlock[2, 3], 
	getBlock[4, 5],
	getBlock[6, 7],
	getBlock2[0, 4096],
	getBlock2[1, 4096],
	{getFC["2", 1000],SoftmaxLayer[]}
},
	"Input" -> encoder, "Output" -> decoder
]


(* ::Subchapter:: *)
(*Export Model*)


Export["imagenet_vgg11.WXF", mainNet]
