(* ::Package:: *)

(* ::Subchapter:: *)
(*Import Weights*)


SetDirectory@NotebookDirectory[];
Clear["Global`*"];
<< DeepMath`;
DeepMath`NetMerge;
$name = "ComixGAN Comic Style Transfer Alpha";
params = Import[$name <> ".hdf5", "Data"];


(* ::Subchapter:: *)
(*Pre-defined Structure*)


$NCHW = TransposeLayer[{1<->4, 2<->3, 3<->4}];
getName[s_] := TemplateApply["/`1`/`1`", {s}];
getPad[n_] := PaddingLayer[{{0, 0}, {n, n}, {n, n}}, Padding -> "Reflected"];
getCW[name_, s_, p_, k_ : 1] := ConvolutionLayer[
	"Weights" -> k * $NCHW@params[getName[name] <> "/kernel:0"],
	"Biases" -> None, "Stride" -> s, "PaddingSize" -> p
];
getCN[name_, s_, p_, k_ : 1] := ConvolutionLayer[
	"Weights" -> k * $NCHW@params[getName[name] <> "/kernel:0"],
	"Biases" -> k * params[getName[name] <> "/bias:0"],
	"Stride" -> s, "PaddingSize" -> p
];
getDN[name_, s_, p_] := DeconvolutionLayer[
	"Weights" -> $NCHW@params[getName[name] <> "/kernel:0"],
	"Biases" -> params[getName[name] <> "/bias:0"],
	"Stride" -> s, "PaddingSize" -> p
];
getIN[name_] := NormalizationLayer[
	"Biases" -> params[getName[name] <> "/beta:0"],
	"Scaling" -> params[getName[name] <> "/gamma:0"],
	"Epsilon" -> 0.001
];


getBlock[i_] := GeneralUtilities`Scope[
	path = NetChain@{
		getCW["conv2d_" <> ToString[i], 1, 1],
		getIN["instance_normalization_" <> ToString[i - 6]],
		Ramp,
		getCW["conv2d_" <> ToString[i + 1], 1, 1],
		getIN["instance_normalization_" <> ToString[i - 5]]
	};
	NetMerge[path, Plus, Expand -> True]
];


(* ::Subchapter:: *)
(*Main*)


mainNet = NetChain[{
	{
		getCW["conv2d_8", 1, 3],
		getIN["instance_normalization_4"],
		Ramp
	},
	{
		getCN["conv2d_9", 2, 1],
		getCW["conv2d_10", 1, 1],
		getIN["instance_normalization_5"],
		Ramp
	},
	{
		getCN["conv2d_11", 2, 1],
		getCW["conv2d_12", 1, 1],
		getIN["instance_normalization_6"],
		Ramp
	},
	NetChain@Table[getBlock[i], {i, 13, 27, 2}],
	{
		ResizeLayer[Scaled /@ {2, 2}],
		getCW["conv2d_29", 1, 1],
		getIN["instance_normalization_23"],
		Ramp
	},
	{
		ResizeLayer[Scaled /@ {2, 2}],
		getCW["conv2d_30", 1, 1],
		getIN["instance_normalization_24"],
		Ramp
	},
	getCN["conv2d_31", 1, 3],
	LogisticSigmoid
},
	"Output" -> "Image"
]


(* ::Subchapter:: *)
(*Testing*)


img = ExampleData[{"TestImage", "House"}]
newNet = NetReplacePart[mainNet, "Input" -> NetEncoder[{"Image", ImageDimensions@img}]];
newNet[img, TargetDevice -> "GPU"]


(* ::Subchapter:: *)
(*Export Model*)


Export[$name <> ".MAT", mainNet, "WXF", PerformanceGoal -> "Speed"]
