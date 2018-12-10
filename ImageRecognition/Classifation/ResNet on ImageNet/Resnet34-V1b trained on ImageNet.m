(* ::Package:: *)

SetDirectory@NotebookDirectory[];
<< NeuralNetworks`
<< MXNetLink`
<< DeepMath`
DateString[]


(* ::Subitem:: *)
(*Tue 11 Dec 2018 00:44:07*)


(* ::Subchapter:: *)
(*Import Weights*)


params = NDArrayImport["imagenet_resnet34_v1b-0000.params"];


(* ::Subchapter:: *)
(*Encoder & Decoder*)


mShift = {0.485, 0.456, 0.406};
vShift = {0.229, 0.224, 0.225}^2;
encoder = NetEncoder[{"Image", 224, "MeanImage" -> mShift, "VarianceImage" -> vShift}]
decoder = NetExtract[NetModel["ResNet-50 Trained on ImageNet Competition Data"], "Output"]


(* ::Subchapter::Closed:: *)
(*Pre-defined Structure*)


getCN[name_String, p_ : 1, s_ : 1] := ConvolutionLayer[
	"Weights" -> params["arg:resnetv1b_" <> name <> "_weight"],
	"Biases" -> None, "PaddingSize" -> p, "Stride" -> s
];
getBN[name_String] := BatchNormalizationLayer[
	"Epsilon" -> 1*^-5,
	"Beta" -> params["arg:resnetv1b_" <> name <> "_beta"],
	"Gamma" -> params["arg:resnetv1b_" <> name <> "_gamma"],
	"MovingMean" -> params["aux:resnetv1b_" <> name <> "_running_mean"],
	"MovingVariance" -> params["aux:resnetv1b_" <> name <> "_running_var"]
];
getBlock2[num_, j_ : 2] := Block[
	{i = ToString@num, path1, path2},
	path1 = NetChain2Graph@NetChain[{
		getCN["layers" <> i <> "_conv0", 1, j],
		getBN["layers" <> i <> "_batchnorm0"],
		ElementwiseLayer["ReLU"],
		getCN["layers" <> i <> "_conv1", 1, 1],
		getBN["layers" <> i <> "_batchnorm1"]
	}];
	path2 = NetChain2Graph@NetChain[{
		getCN["down" <> i <> "_conv0", 0, j],
		getBN["down" <> i <> "_batchnorm0"]
	}];
	NetFlatten@NetGraph[{NetMerge[{path1, path2}], Ramp}, {1 -> 2}]
];
getBlock3[num_, j_] := Block[
	{i = ToString@num, path},
	path = NetChain[{
		getCN["layers" <> i <> "_conv" <> ToString[j], 1, 1],
		getBN["layers" <> i <> "_batchnorm" <> ToString[j]],
		ElementwiseLayer["ReLU"],
		getCN["layers" <> i <> "_conv" <> ToString[j + 1], 1, 1],
		getBN["layers" <> i <> "_batchnorm" <> ToString[j + 1]],
		ElementwiseLayer["ReLU"]
	}];
	NetFlatten@NetGraph[{NetMerge@path, Ramp}, {1 -> 2}]
];


(* ::Subchapter:: *)
(*Main*)


extractor = NetChain[{
	{
		getCN["conv0", 3, 2],
		getBN["batchnorm0"],
		ElementwiseLayer["ReLU"]
	},
	PoolingLayer[{3, 3}, "Stride" -> 2, "PaddingSize" -> 1],
	Table[getBlock3[1, i], {i, 0, 4, 2}],
	getBlock2[2, 2],
	Table[getBlock3[2, i], {i, 2, 6, 2}],
	getBlock2[3, 2],
	Table[getBlock3[3, i], {i, 2, 10, 2}],
	getBlock2[4, 2],
	Table[getBlock3[4, i], {i, 2, 4, 2}],
	AggregationLayer[Mean]
}];
classifier = LinearLayer[1000,
	"Weights" -> params["arg:resnetv1b_dense0_weight"],
	"Biases" -> params["arg:resnetv1b_dense0_bias"]
];
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


Export["Resnet34-V1b trained on ImageNet.WXF", mainNet]
