(* ::Package:: *)

SetDirectory@NotebookDirectory[];
Needs["MXNetLink`"]
Needs["NeuralNetworks`"]
DateString[]


(* ::Subitem:: *)
(*Sat 3 Nov 2018 22:22:07*)


(* ::Subchapter:: *)
(*Import Weights*)


raw = Import["DeblurSHC19ConvLayers.hdf5", "Data"];
params[name_String] := Block[
	{prefix, input},
	prefix = "/model_weights/model_1/";
	input = raw[prefix <> name];
	Switch[
		Length@Dimensions@input,
		1, RawArray["Real32", input],
		4, RawArray["Real32", TransposeLayer[{1<->4, 2<->3, 3<->4}][input]],
		_, RawArray["Real32", input]
	]
]


(* ::Subchapter:: *)
(*Encoder & Decoder*)


mShift = {0, 0, 0};
vShift = {1, 1, 1}^2;
encoder = NetEncoder[{"Image", {640, 360}, ColorSpace -> "Grayscale"}]
decoder = NetDecoder["Image"]


(* ::Subchapter::Closed:: *)
(*Pre-defined Structure*)


getCN[i_Integer, p_Integer, s_Integer] := ConvolutionLayer[
	"Weights" -> params["conv2d_" <> ToString[i] <> "/kernel:0"],
	"Biases" -> params["conv2d_" <> ToString[i] <> "/bias:0"],
	"PaddingSize" -> p, "Stride" -> s
];
getBN[i_Integer] := BatchNormalizationLayer[
	"Momentum" -> 0.99,
	"Beta" -> params["batch_normalization_" <> ToString[i] <> "/beta:0"],
	"Gamma" -> params["batch_normalization_" <> ToString[i] <> "/gamma:0"],
	"MovingMean" -> params["batch_normalization_" <> ToString[i] <> "/moving_mean:0"],
	"MovingVariance" -> params["batch_normalization_" <> ToString[i] <> "/moving_variance:0"]
];
getBlock[i_] := NetChain[{
	getBN[i],
	ElementwiseLayer["ReLU"],
	getCN[i + 1, 1, 1]
}];


$body = NetGraph[
	Flatten@{
		getCN[3, 1, 1],
		Table[{getBlock[j], ThreadingLayer[Plus]}, {j, 3, 17}]
	},
	Join[
		{NetPort["Input"] -> 1 -> 2},
		Table[{NetPort["Input"], j} -> j + 1 -> j + 2, {j, 2, 28, 2}],
		{{NetPort["Input"], 30} -> 31}
	]
];


(* ::Subchapter:: *)
(*Main*)


mainNet = NetChain[{
	PaddingLayer[{{0, 0}, {12, 12}, {12, 12}}, "Padding" -> 1],
	getCN[1, 0, 1],
	getBN[1],
	ElementwiseLayer["ReLU"],
	getCN[2, 1, 1],
	getBN[2],
	ElementwiseLayer["ReLU"],
	$body,
	getBN[18],
	ElementwiseLayer["ReLU"],
	getCN[19, 1, 1]
},
	"Input" -> encoder,
	"Output" -> decoder
]


(* ::Subchapter:: *)
(*Export Model*)


Export["SHC Text Deblur trained on Custom.WXF", mainNet]
