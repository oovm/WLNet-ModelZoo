(* ::Package:: *)

SetDirectory@NotebookDirectory[];
<< NeuralNetworks`
<< MXNetLink`
<< DeepMath`
DateString[]


(* ::Subitem:: *)
(*Fri 21 Dec 2018 15:18:10*)


(* ::Subchapter:: *)
(*Import Weights*)


params = NDArrayImport["model-0000.params"];


(* ::Subchapter:: *)
(*Encoder & Decoder*)


encoder = NetEncoder[{"Image", 112}]
decoder = NetDecoder[{"Class", {"Man", "Woman"}}]


(* ::Subchapter:: *)
(*Pre-defined Structure*)


ReLU = ElementwiseLayer["ReLU"];
getBlock[ii_, j_] := Block[
	{i = ToString@ii, path},
	path = NetChain2Graph@NetChain[{
		getCN["stage" <> i <> "_conv" <> ToString[j], 1, 1],
		getBN["stage" <> i <> "_batchnorm" <> ToString[j]],
		ReLU,
		getCN["stage" <> i <> "_conv" <> ToString[j + 1], 1, 1],
		getBN["stage" <> i <> "_batchnorm" <> ToString[j + 1]]
	}];
	NetFlatten@NetGraph[{NetMerge@path, ReLU}, {1 -> 2}]
]
getBlock2[ii_] := Block[
	{i = ToString@ii, lhs, rhs},
	lhs = NetChain2Graph@NetChain[{
		getCN["stage" <> i <> "_conv2", 0, 2],
		getBN["stage" <> i <> "_batchnorm2"]
	}];
	rhs = NetChain2Graph@NetChain[{
		getCN["stage" <> i <> "_conv0", 1, 1],
		getBN["stage" <> i <> "_batchnorm0"],
		ReLU,
		getCN["stage" <> i <> "_conv1", 1, 2],
		getBN["stage" <> i <> "_batchnorm1"]
	}];
	NetFlatten@NetGraph[{NetMerge[{lhs, rhs}], ReLU}, {1 -> 2}]
]


head = {
	BatchNormalizationLayer[
		"Epsilon" -> 1*^-5,
		"Beta" -> params["arg:gender_resnet0_batchnorm0" <> "_beta"],
		"Gamma" -> params["arg:gender_resnet0_batchnorm0" <> "_gamma"],
		"MovingMean" -> Normal@params["aux:gender_resnet0_batchnorm0" <> "_running_mean"] / 255,
		"MovingVariance" -> Normal@params["aux:gender_resnet0_batchnorm0" <> "_running_var"] / 255^2
	],
	getCN["conv0", 1, 1],
	getBN["batchnorm1"],
	ReLU
};
tail = {
	BatchNormalizationLayer[
		"Epsilon" -> 1*^-5,
		"Beta" -> params["arg:gender_embeddingblock0_batchnorm0" <> "_beta"],
		"Gamma" -> params["arg:gender_embeddingblock0_batchnorm0" <> "_gamma"],
		"MovingMean" -> params["aux:gender_embeddingblock0_batchnorm0" <> "_running_mean"],
		"MovingVariance" -> params["aux:gender_embeddingblock0_batchnorm0" <> "_running_var"]
	],
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
}]
classifier = {
	LinearLayer[2,
		"Weights" -> params["arg:gender_dense0" <> "_weight"],
		"Biases" -> params["arg:gender_dense0" <> "_bias"]
	]
};
mainNet = NetChain[{
	"Extractor" -> extractor,
	"Classifier" -> classifier,
	"Predictor" -> SoftmaxLayer[]
},
	"Input" -> encoder,
	"Output" -> decoder
]


(* ::Subchapter:: *)
(*Export Model*)


Export["ResNet50-Gender trained on AsianFace.WXF", mainNet]
