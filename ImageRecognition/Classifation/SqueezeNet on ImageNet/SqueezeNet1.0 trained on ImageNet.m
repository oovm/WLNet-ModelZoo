(* ::Package:: *)

SetDirectory@NotebookDirectory[];
Needs["MXNetLink`"]
Needs["NeuralNetworks`"]
DateString[]


(* ::Subitem:: *)
(*Fri 26 Oct 2018 17:28:43*)


(* ::Subchapter:: *)
(*Import Weights*)


params = NDArrayImport["squeezenet1.0-0000.params"];


(* ::Subchapter:: *)
(*Encoder & Decoder*)


mShift = {0.485, 0.456, 0.406};
vShift = {0.229, 0.224, 0.225}^2;
encoder = NetEncoder[{"Image", 227, "MeanImage" -> mShift, "VarianceImage" -> vShift}]
decoder = NetExtract[NetModel["ResNet-50 Trained on ImageNet Competition Data"], "Output"]


(* ::Subchapter::Closed:: *)
(*Pre-defined Structure*)


getCN[i_, p_, s_] := ConvolutionLayer[
	"Weights" -> params["arg:squeezenet0_conv" <> ToString[ i ] <> "_weight"],
	"Biases" -> params["arg:squeezenet0_conv" <> ToString[i] <> "_bias"],
	"PaddingSize" -> p, "Stride" -> s
]
getBlock[i_] := NetGraph[{
	NetGraph[{getCN[i, 0, 1], Ramp}, {1 -> 2}],
	NetGraph[{getCN[i + 1, 0, 1], Ramp}, {1 -> 2}],
	NetGraph[{getCN[i + 2, 1, 1], Ramp}, {1 -> 2}],
	CatenateLayer[]
},
	{NetPort["Input"] -> 1 -> {2, 3} -> 4}
] // NetFlatten


(* ::Subchapter:: *)
(*Main*)


mainNet = NetChain[{
	getCN[0, 0, 2],
	ElementwiseLayer["ReLU"],
	PoolingLayer[{3, 3}, "Stride" -> 2],
	NetChain@Table[getBlock[i], {i, 1, 7, 3}],
	PoolingLayer[{3, 3}, "Stride" -> 2],
	NetChain@Table[getBlock[i], {i, 10, 19, 3}],
	PoolingLayer[{3, 3}, "Stride" -> 2],
	NetChain@Table[getBlock[i], {i, 22, 22, 3}],
	DropoutLayer[0.5],
	getCN[25, 0, 1],
	ElementwiseLayer["ReLU"],
	PoolingLayer[{13, 13}, "Stride" -> 13, "Function" -> Mean],
	FlattenLayer[],
	SoftmaxLayer[]
},
	"Input" -> encoder,
	"Output" -> decoder
]


(* ::Subchapter:: *)
(*Export Model*)


Export["SqueezeNet1.0 trained on ImageNet.WXF", mainNet]
