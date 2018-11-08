(* ::Package:: *)

SetDirectory@NotebookDirectory[];
<< NeuralNetworks`
<< MXNetLink`
<< DeepMath`
DateString[]


(* ::Subitem:: *)
(*Thu 8 Nov 2018 19:05:21*)


(* ::Subchapter:: *)
(*Import Weights*)


raw = Import["generator.h5", "Data"];
params[name_String] := Block[
	{prefix, input},
	prefix = StringJoin["/", First@StringSplit[name, "/"], "/"];
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
encoder = NetEncoder[{"Image", {640, 360}}]
decoder = NetDecoder["Image"]


(* ::Subchapter:: *)
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
getBlock[i_] := Block[
	{path},
	path = NetChain[{
		PaddingLayer[Partition[{0, 0, 1, 1, 1, 1}, 2], "Padding" -> "Reflected"],
		getCN[i, 0, 1],
		getBN[i],
		ElementwiseLayer["ReLU"],
		DropoutLayer[0.5],
		PaddingLayer[Partition[{0, 0, 1, 1, 1, 1}, 2], "Padding" -> "Reflected"],
		getCN[i + 1, 0, 1],
		getBN[i + 1]
	}];
	NetMerge[path, Expand -> All]
];
getBlock2[i_] := NetChain[{
	ResizeLayer[Scaled /@ {2, 2}, Resampling -> "Nearest"],
	getCN[i, 1, 1],
	getBN[i],
	ElementwiseLayer["ReLU"]
}];


(* ::Subchapter:: *)
(*Main*)


mainNet = NetChain[{
	PaddingLayer[Partition[{0, 0, 3, 3, 3, 3}, 2], "Padding" -> "Reflected"],
	getCN[1, 0, 1],
	getBN[1],
	ElementwiseLayer["ReLU"],
	{getCN[2, 1, 2], getBN[2], ElementwiseLayer["ReLU"]},
	{getCN[3, 1, 2], getBN[3], ElementwiseLayer["ReLU"]},
	NetChain@Table[getBlock[i], {i, 4, 20, 2}],
	getBlock2[22],
	getBlock2[23],
	PaddingLayer[Partition[{0, 0, 3, 3, 3, 3}, 2], "Padding" -> "Reflected"],
	getCN[24, 0, 1],
	Tanh
},
	"Input" -> encoder,
	"Output" -> decoder
]


(* ::Subchapter:: *)
(*Export Model*)


Export["DeblurGAN trained on GOPRO.WXF", mainNet]
