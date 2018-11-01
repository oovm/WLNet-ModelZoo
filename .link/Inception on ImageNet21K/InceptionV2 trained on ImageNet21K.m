(* ::Package:: *)

SetDirectory@NotebookDirectory[];
<< NeuralNetworks`
<< MXNetLink`
<< DeepMath`
DateString[]


(* ::Subitem:: *)
(*Thu 1 Nov 2018 11:25:25*)


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
];
getBN[name_String] := BatchNormalizationLayer[
	"Epsilon" -> 1*^-5,
	"Beta" -> params["arg:bn_" <> name <> "_beta"],
	"Gamma" -> params["arg:bn_" <> name <> "_gamma"],
	"MovingMean" -> params["aux:bn_" <> name <> "_moving_mean"],
	"MovingVariance" -> params["aux:bn_" <> name <> "_moving_var"]
];


getBlock[name_, p_, s_] := NetChain@{
	getCN[name, p, s],
	getBN[name] ,
	ElementwiseLayer["ReLU"]
};
getBlock2[name_String] := Block[
	{path1, path2, path3, path4},
	path1 = getBlock[name <> "_1x1", 0, 1];
	path2 = NetChain@{
		getBlock[name <> "_3x3_reduce", 0, 1],
		getBlock[name <> "_3x3", 1, 1]
	} // NetFlatten;
	path3 = NetChain@{
		getBlock[name <> "_double_3x3_reduce", 0, 1],
		getBlock[name <> "_double_3x3_0", 1, 1],
		getBlock[name <> "_double_3x3_1", 1, 1]
	} // NetFlatten;
	path4 = NetChain@{
		PoolingLayer[{3, 3}, "PaddingSize" -> 1, "Function" -> Mean],
		getBlock[name <> "_proj", 0, 1]
	} // NetFlatten;
	NetMerge[{path1, path2, path3, path4}, Join, Expand -> All]
];
getBlock3[name_String] := Block[
	{path1, path2, path3},
	path1 = NetChain@{
		getBlock[name <> "_3x3_reduce", 0, 1],
		getBlock[name <> "_3x3", 1, 2]
	} // NetFlatten;
	path2 = NetChain@{
		getBlock[name <> "_double_3x3_reduce", 0, 1],
		getBlock[name <> "_double_3x3_0", 1, 1],
		getBlock[name <> "_double_3x3_1", 1, 2]
	} // NetFlatten;
	path3 = PoolingLayer[{3, 3}, "PaddingSize" -> 1, "Stride" -> 2];
	NetMerge[{path1, path2, path3}, Join, Expand -> All]
];


(* ::Subchapter:: *)
(*Main*)


mainNet = NetChain[{
	getBlock["conv1", 3, 2],
	PoolingLayer[{3, 3}, "Stride" -> 2],
	getBlock["conv2red", 0, 1],
	getBlock["conv2", 1, 1],
	PoolingLayer[{3, 3}, "Stride" -> 2],
	NetChain@{getBlock2["3a"], getBlock2["3b"]},
	getBlock3["3c"],
	NetChain@{getBlock2["4a"], getBlock2["4b"], getBlock2["4c"], getBlock2["4d"]},
	getBlock3["4e"],
	NetChain@{getBlock2["5a"], getBlock2["5b"]},
	AggregationLayer[Mean],
	{
		LinearLayer[
			"Weights" -> params["arg:fc1_weight"],
			"Biases" -> params["arg:fc1_bias"]
		],
		SoftmaxLayer[]
	}
},
	"Input" -> encoder,
	"Output" -> decoder
]


(* ::Subchapter:: *)
(*Export Model*)


Export["InceptionV2 trained on ImageNet21K.WXF", mainNet]
