(* ::Package:: *)

SetDirectory@NotebookDirectory[];
Needs["MXNetLink`"]
Needs["NeuralNetworks`"]
DateString[]


(* ::Subitem:: *)
(*Thu 25 Oct 2018 13:10:02*)


(* ::Subchapter:: *)
(*Import Weights*)


params = NDArrayImport["EDSR_x4-0000.params"];


(* ::Subchapter:: *)
(*Encoder & Decoder*)


encoder = NetEncoder[{"Image", {640, 360}}]
decoder = NetDecoder["Image"]

"Use this encoder if not use shift convolution";
NetEncoder[{
	"Image", {640, 360},
	"MeanImage" -> {0.4488, 0.4371, 0.4040},
	"VarianceImage" -> 1 / 255^2
}];


(* ::Subchapter::Closed:: *)
(*Pre-defined Structure*)


getBlock[i_] := NetGraph[{
	ConvolutionLayer[
		"Weights" -> params["arg:body." <> ToString[i] <> ".body.0.weight"],
		"Biases" -> params["arg:body." <> ToString[i] <> ".body.0.bias"],
		"PaddingSize" -> 1, "Stride" -> 1
	],
	ElementwiseLayer["ReLU"],
	ConvolutionLayer[
		"Weights" -> params["arg:body." <> ToString[i] <> ".body.2.weight"],
		"Biases" -> params["arg:body." <> ToString[i] <> ".body.2.bias"],
		"PaddingSize" -> 1, "Stride" -> 1
	],
	ThreadingLayer[#1 + 0.1#2&]
}, {
	NetPort["Input"] -> 1 -> 2 -> 3,
	{NetPort["Input"], 3} -> 4
}];
$body = NetGraph[{
	NetChain@Array[getBlock, 31],
	ConvolutionLayer[
		"Weights" -> params["arg:body.32.weight"],
		"Biases" -> params["arg:body.32.bias"],
		"PaddingSize" -> 1, "Stride" -> 1
	],
	ThreadingLayer[Plus]
}, {
	NetPort["Input"] -> 1 -> 2,
	{NetPort["Input"], 2} -> 3
}];


(* ::Subchapter:: *)
(*Main*)


mainNet = NetChain[{
	ConvolutionLayer[
		"Weights" -> params["arg:sub_mean.weight"],
		"Biases" -> params["arg:sub_mean.bias"],
		"PaddingSize" -> 0, "Stride" -> 1
	],
	ConvolutionLayer[
		"Weights" -> params["arg:head.0.weight"],
		"Biases" -> params["arg:head.0.bias"],
		"PaddingSize" -> 1, "Stride" -> 1
	],
	$body,
	ConvolutionLayer[
		"Weights" -> params["arg:tail.0.0.weight"],
		"Biases" -> params["arg:tail.0.0.bias"],
		"PaddingSize" -> 1, "Stride" -> 1
	],
	PixelShuffleLayer[2],
	ConvolutionLayer[
		"Weights" -> params["arg:tail.0.2.weight"],
		"Biases" -> params["arg:tail.0.2.bias"],
		"PaddingSize" -> 1, "Stride" -> 1
	],
	PixelShuffleLayer[2],
	ConvolutionLayer[
		"Weights" -> params["arg:tail.1.weight"],
		"Biases" -> params["arg:tail.1.bias"],
		"PaddingSize" -> 1, "Stride" -> 1
	],
	ConvolutionLayer[
		"Weights" -> params["arg:add_mean.weight"],
		"Biases" -> params["arg:add_mean.bias"],
		"PaddingSize" -> 0, "Stride" -> 1
	]
},
	"Input" -> encoder,
	"Output" -> decoder
]


(* ::Subchapter:: *)
(*Export Model*)


Export["EDSR4x trained on DIV2K.WXF", mainNet]
