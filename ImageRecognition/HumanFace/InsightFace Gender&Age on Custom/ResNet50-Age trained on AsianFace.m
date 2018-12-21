(* ::Package:: *)

SetDirectory@NotebookDirectory[];
<< NeuralNetworks`
<< MXNetLink`
<< DeepMath`
DateString[]


(* ::Subitem:: *)
(*Fri 21 Dec 2018 15:10:07*)


(* ::Subchapter:: *)
(*Import Weights*)


params = NDArrayImport["model-0000.params"];


(* ::Subchapter:: *)
(*Encoder & Decoder*)


encoder = NetEncoder[{"Image", 112}]


(* ::Subchapter::Closed:: *)
(*Pre-defined Structure*)


prefix = "age";
ReLU = ElementwiseLayer["ReLU"];
getCN[name_String, p_ : 1, s_ : 1] := ConvolutionLayer[
	"Weights" -> params["arg:" <> prefix <> "_" <> name <> "_weight"],
	"Biases" -> None, "PaddingSize" -> p, "Stride" -> s
];
getBN[name_String] := BatchNormalizationLayer[
	"Epsilon" -> 1*^-5,
	"Beta" -> params["arg:" <> prefix <> "_" <> name <> "_beta"],
	"Gamma" -> params["arg:" <> prefix <> "_" <> name <> "_gamma"],
	"MovingMean" -> params["aux:" <> prefix <> "_" <> name <> "_running_mean"],
	"MovingVariance" -> params["aux:" <> prefix <> "_" <> name <> "_running_var"]
];
getBlock[ii_, j_] := Block[
	{i = ToString@ii, path},
	path = NetChain2Graph@NetChain[{
		getCN["resnet0_stage" <> i <> "_conv" <> ToString[j], 1, 1],
		getBN["resnet0_stage" <> i <> "_batchnorm" <> ToString[j]],
		ReLU,
		getCN["resnet0_stage" <> i <> "_conv" <> ToString[j + 1], 1, 1],
		getBN["resnet0_stage" <> i <> "_batchnorm" <> ToString[j + 1]]
	}];
	NetFlatten@NetGraph[{NetMerge@path, ReLU}, {1 -> 2}]
];
getBlock2[ii_] := Block[
	{i = ToString@ii, lhs, rhs},
	lhs = NetChain2Graph@NetChain[{
		getCN["resnet0_stage" <> i <> "_conv2", 0, 2],
		getBN["resnet0_stage" <> i <> "_batchnorm2"]
	}];
	rhs = NetChain2Graph@NetChain[{
		getCN["resnet0_stage" <> i <> "_conv0", 1, 1],
		getBN["resnet0_stage" <> i <> "_batchnorm0"],
		ReLU,
		getCN["resnet0_stage" <> i <> "_conv1", 1, 2],
		getBN["resnet0_stage" <> i <> "_batchnorm1"]
	}];
	NetFlatten@NetGraph[{NetMerge[{lhs, rhs}], ReLU}, {1 -> 2}]
];


head = {
	BatchNormalizationLayer[
		"Epsilon" -> 1*^-5,
		"Beta" -> params["arg:age_resnet0_batchnorm0" <> "_beta"],
		"Gamma" -> params["arg:age_resnet0_batchnorm0" <> "_gamma"],
		"MovingMean" -> Normal@params["aux:age_resnet0_batchnorm0" <> "_running_mean"] / 255,
		"MovingVariance" -> Normal@params["aux:age_resnet0_batchnorm0" <> "_running_var"] / 255^2
	],
	getCN["resnet0_conv0", 1, 1],
	getBN["resnet0_batchnorm1"],
	ReLU
};
tail = {
	getBN["embeddingblock0_batchnorm0"],
	ReLU,
	AggregationLayer@Mean
};


(* ::Subchapter:: *)
(*Main*)


extractor = NetChain[{
	head,
	getBlock2[1],
	Table[getBlock[1, i], {i, 3, 5, 2}],
	getBlock2[2],
	Table[getBlock[2, i], {i, 3, 7, 2}],
	getBlock2[3],
	Table[getBlock[3, i], {i, 3, 27, 2}],
	getBlock2[4],
	Table[getBlock[4, i], {i, 3, 5, 2}],
	tail
}];
classifier = {
	LinearLayer[200,
		"Weights" -> params["arg:" <> prefix <> "_dense0" <> "_weight"],
		"Biases" -> params["arg:" <> prefix <> "_dense0" <> "_bias"]
	],
	ReshapeLayer[{100, 2}]
};


mainNet = NetChain[{
	"Extractor" -> extractor,
	"Classifier" -> classifier
},
	"Input" -> encoder
]


(* ::Text:: *)
(*ans = mainNet[img, TargetDevice -> "GPU"]*)
(*Tr@Boole[Less@@@ans]*)
(*ListLinePlot@Transpose@ans*)


(* ::Subchapter:: *)
(*Export Model*)


Export["ResNet50-" <> Capitalize@prefix <> " trained on AsianFace.WXF", mainNet]
