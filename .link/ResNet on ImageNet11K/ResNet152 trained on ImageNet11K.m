(* ::Package:: *)

SetDirectory@NotebookDirectory[];
<< NeuralNetworks`
<< MXNetLink`
<< DeepMath`
DateString[]


(* ::Subitem:: *)
(*Thu 1 Nov 2018 13:11:48*)


(* ::Subchapter:: *)
(*Import Weights*)


params = NDArrayImport["resnet-152-0000.params"];


(* ::Subchapter:: *)
(*Encoder & Decoder*)


mShift = Normal@ params["aux:bn_data_moving_mean"] / 255;
vShift = Normal@ params["aux:bn_data_moving_var"] / 255^2;
filter[line_] := Block[
	{tag = StringSplit[First@StringSplit[line, ","], " "]},
	StringJoin@{First[tag], "::", StringRiffle[Rest[tag], " "]}
];
tags = filter /@ Import["synset.txt", "List"];
encoder = NetEncoder[{"Image", 224, "MeanImage" -> mShift, "VarianceImage" -> vShift}]
decoder = NetDecoder[{"Class", tags}]


(* ::Subchapter:: *)
(*Pre-defined Structure*)


getCN[name_String, p_Integer, s_Integer] := ConvolutionLayer[
	"Weights" -> params["arg:" <> name <> "_weight"],
	"Biases" -> None, "PaddingSize" -> p, "Stride" -> s
];
getBN[name_String] := BatchNormalizationLayer[
	"Epsilon" -> 2*^-5,
	"Beta" -> params["arg:" <> name <> "_beta"],
	"Gamma" -> params["arg:" <> name <> "_gamma"],
	"MovingMean" -> params["aux:" <> name <> "_moving_mean"],
	"MovingVariance" -> params["aux:" <> name <> "_moving_var"]
];



getBlock[name_String, p_, s_] := Block[
	{head, path1, path2},
	head = NetChain@{
		getBN[name <> "_unit1_bn1"] ,
		ElementwiseLayer["ReLU"]
	};
	path1 = NetChain@{
		getCN[name <> "_unit1_conv1", 0, 1],
		getBN[name <> "_unit1_bn2"] ,
		ElementwiseLayer["ReLU"],
		getCN[name <> "_unit1_conv2", p, s],
		getBN[name <> "_unit1_bn3"] ,
		ElementwiseLayer["ReLU"],
		getCN[name <> "_unit1_conv3", 0, 1]
	};
	path2 = getCN[name <> "_unit1_sc", 0, s];
	NetChain2Graph /@ NetGraph[
		{head, path1, path2, ThreadingLayer[Plus]},
		{NetPort["Input"] -> 1 -> {2, 3} -> 4}
	]
];
getBlock2[i_, j_, k_, p_, s_] := NetChain@{
	getBN[StringRiffle[{"stage", i, "_unit", j, "_bn", k}, ""]] ,
	ElementwiseLayer["ReLU"],
	getCN[StringRiffle[{"stage", i, "_unit", j, "_conv", k}, ""], p, s]
};
getBlock3[i_, j_] := Block[
	{path},
	path = NetChain@{
		getBlock2[i, j, 1, 0, 1],
		getBlock2[i, j, 2, 1, 1],
		getBlock2[i, j, 3, 0, 1]
	} // NetFlatten;
	NetMerge[path, Plus, Expand -> All]
];


$head = NetChain@{
	getCN["conv0", 3, 2],
	getBN["bn0"] ,
	ElementwiseLayer["ReLU"],
	PoolingLayer[{3, 3}, "PaddingSize" -> 1, "Stride" -> 2]
};


(* ::Subchapter:: *)
(*Main*)


mainNet = NetChain[{
	$head,
	getBlock["stage1", 1, 1],
	NetChain@Table[getBlock3[1, j], {j, 2, 3}],
	getBlock["stage2", 1, 2],
	NetChain@Table[getBlock3[2, j], {j, 2, 8}],
	getBlock["stage3", 1, 2],
	NetChain@Table[getBlock3[3, j], {j, 2, 36}],
	getBlock["stage4", 1, 2],
	NetChain@Table[getBlock3[4, j], {j, 2, 3}],
	getBN["bn1"] ,
	ElementwiseLayer["ReLU"],
	AggregationLayer[Mean],
	LinearLayer[
		"Weights" -> params["arg:fc1_weight"],
		"Biases" -> params["arg:fc1_bias"]
	],
	SoftmaxLayer[]
},
	"Input" -> encoder,
	"Output" -> decoder
]


(* ::Subchapter:: *)
(*Export Model*)


Export["ResNet152 trained on ImageNet11K.WXF", mainNet]
