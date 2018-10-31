(* ::Package:: *)

SetDirectory@NotebookDirectory[];
Needs["MXNetLink`"]
Needs["NeuralNetworks`"]
DateString[]


(* ::Subitem:: *)
(*Thu 1 Nov 2018 00:11:42*)


(* ::Subchapter:: *)
(*Import Weights*)


params = NDArrayImport["Inception-0009.params"];


(* ::Subchapter:: *)
(*Encoder & Decoder*)


mShift = {117, 117, 117} / 255;
vShift = {0, 0, 0}^2;
fliter[line_] := Block[
	{tag = StringSplit[First@StringSplit[line, ","], " "]},
	StringJoin@{First[tag], "::", StringRiffle[Rest[tag], " "]}
]
tags = Sort[fliter /@ Import["synset.txt", "List"]];
encoder = NetEncoder[{"Image", 224, "MeanImage" -> mShift}]
decoder = NetDecoder[{"Class", tags}]


(* ::Subchapter:: *)
(*Pre-defined Structure*)


getCN[name_String, p_Integer, s_Integer] := ConvolutionLayer[
	"Weights" -> params["arg:conv_" <> name <> "_weight"],
	"Biases" -> params["arg:conv_" <> name <> "_bias"],
	"PaddingSize" -> p, "Stride" -> s
]
getBN[name_String] := BatchNormalizationLayer[
	"Epsilon" -> 1*^-5,
	"Beta" -> params["arg:bn_" <> name <> "_beta"],
	"Gamma" -> params["arg:bn_" <> name <> "_gamma"],
	"MovingMean" -> params["aux:bn_" <> name <> "_moving_mean"],
	"MovingVariance" -> params["aux:bn_" <> name <> "_moving_var"]
]


$head = NetChain@{
	getCN["conv1", 3, 2],
	getBN["conv1"] ,
	ElementwiseLayer["ReLU"],
	PoolingLayer[{3, 3}, "Stride" -> 2],
	getCN["conv2red", 3, 2],
	getBN["conv2red"] ,
	ElementwiseLayer["ReLU"],
	getCN["conv2", 3, 2],
	getBN["conv2"] ,
	ElementwiseLayer["ReLU"],
	PoolingLayer[{3, 3}, "Stride" -> 2]
}





getBlock[name_, p_, s_] := NetChain@{
	getCN[name, p, s],
	getBN[name] ,
	ElementwiseLayer["ReLU"]
};
path1 = getBlock["3a_1x1", 0, 1]
path2 = NetChain@{
	getBlock["3a_3x3_reduce", 0, 1],
	getBlock["3a_3x3", 1, 1]
} // NetFlatten
path3 = NetChain@{
	getBlock["3a_double_3x3_reduce", 0, 1],
	getBlock["3a_double_3x3_0", 1, 1],
	getBlock["3a_double_3x3_1", 1, 1]
} // NetFlatten
path4 = NetChain@{
	PoolingLayer[{3, 3}, "PaddingSize" -> 1, "Function" -> Mean],
	getBlock["3a_proj", 0, 1]
} // NetFlatten


(* ::Subchapter:: *)
(*Main*)





(* ::Subchapter:: *)
(*Export Model*)


Export["InceptionV2 trained on ImageNet21K.WXF", mainNet]
