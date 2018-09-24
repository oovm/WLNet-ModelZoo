(* ::Package:: *)
(* ::Title:: *)
(*Tools*)
(* ::Subchapter:: *)
(*Introduce*)
NetChain2Graph::usage = "Transform a NetChain to NetGraph.";
ImageNetEncoder::usage = "";
RemoveLayerShape::usage = "Try to remove the shape of the layer";
(* ::Subchapter:: *)
(*Main*)
(* ::Subsection:: *)
(*Settings*)
Begin["`Tools`"];
Version$Tools = "V0.0";
Updated$Tools = "2018-10-09";
(* ::Subsection::Closed:: *)
(*Codes*)
(* ::Subsubsection:: *)
(*NetChain2Graph*)
NetChain2Graph[other___] := other;
NetChain2Graph[net_NetChain] := Block[
	{nets = Normal@net},
	NetGraph[nets,
		Rule @@@ Partition[Range@Length@nets, 2, 1],
		"Input" -> NetExtract[net, "Input"],
		"Output" -> NetExtract[net, "Output"]
	];
];
(* ::Subsubsection:: *)
(*ImageNetEncoder*)
ImageNetEncoder[size_ : 224, c_ : "RGB"] := NetEncoder[{
	"Image", size,
	ColorSpace -> c,
	"MeanImage" -> {.485, .456, .406},
	"VarianceImage" -> {.229, .224, .225}^2
}];


RemoveLayerShape[layer_ConvolutionLayer] := With[
	{
		k = NetExtract[layer, "OutputChannels"],
		kernelSize = NetExtract[layer, "KernelSize"] ,
		weights = NetExtract[layer, "Weights"],
		biases = NetExtract[layer, "Biases"],
		padding = NetExtract[layer, "PaddingSize"],
		stride = NetExtract[layer, "Stride"],
		dilation = NetExtract[layer, "Dilation"]
	},
	ConvolutionLayer[k, kernelSize,
		"Weights" -> weights, "Biases" -> biases,
		"PaddingSize" -> padding, "Stride" -> stride,
		"Dilation" -> dilation
	]
];

RemoveLayerShape[layer_PoolingLayer] := With[
	{
		f = NetExtract[layer, "Function"],
		kernelSize = NetExtract[layer, "KernelSize"] ,
		padding = NetExtract[layer, "PaddingSize"],
		stride = NetExtract[layer, "Stride"]
	},
	PoolingLayer[kernelSize, stride,
		"PaddingSize" -> padding, "Function" -> f
	]
];

RemoveLayerShape[layer_ElementwiseLayer] := With[
	{f = NetExtract[layer, "Function"]},
	ElementwiseLayer[f]
];
RemoveLayerShape[layer_SoftmaxLayer] := Nothing;
RemoveLayerShape[layer_FlattenLayer] := Nothing;

(* ::Subsection:: *)
(*Additional*)
SetAttributes[
	{ },
	{Protected, ReadProtected}
];
End[]
