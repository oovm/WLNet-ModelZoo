(* ::Package:: *)

SetDirectory@NotebookDirectory[];
Needs["MXNetLink`"]
Needs["NeuralNetworks`"]
DateString[]


(* ::Subitem:: *)
(*Sun 28 Oct 2018 21:38:10*)


(* ::Subchapter:: *)
(*Import Weights*)


params = NDArrayImport["ProSR_2x-0000.params"];


(* ::Subchapter:: *)
(*Encoder & Decoder*)


encoder = NetEncoder[{"Image", {640, 360}}]
decoder = NetDecoder["Image"]


(* ::Subchapter::Closed:: *)
(*Pre-defined Structure*)


getBlock[i_, j_, k_] := NetGraph[{
	ConvolutionLayer[
		"Weights" -> params["arg:pyramid_residual_" <> ToString[i] <> ".residual_denseblock_" <> ToString[j] <> ".dense_block.denselayer" <> ToString[k] <> ".conv_1.weight"],
		"Biases" -> params["arg:pyramid_residual_" <> ToString[i] <> ".residual_denseblock_" <> ToString[j] <> ".dense_block.denselayer" <> ToString[k] <> ".conv_1.bias"],
		"PaddingSize" -> 0, "Stride" -> 1
	],
	ElementwiseLayer["ReLU"],
	PaddingLayer[{{0, 0}, {1, 1}, {1, 1}}, "Padding" -> "Reflected"],
	ConvolutionLayer[
		"Weights" -> params["arg:pyramid_residual_" <> ToString[i] <> ".residual_denseblock_" <> ToString[j] <> ".dense_block.denselayer" <> ToString[k] <> ".conv_2.conv.1.weight"],
		"Biases" -> params["arg:pyramid_residual_" <> ToString[i] <> ".residual_denseblock_" <> ToString[j] <> ".dense_block.denselayer" <> ToString[k] <> ".conv_2.conv.1.bias"],
		"PaddingSize" -> 0, "Stride" -> 1
	],
	CatenateLayer[]
}, {
	NetPort["Input"] -> 1 -> 2 -> 3 -> 4,
	{NetPort["Input"], 4} -> 5
}];
getBlock2[i_, j_] := NetGraph[{
	NetChain@Table[getBlock[i, j, n], {n, 1, 8}],
	ConvolutionLayer[
		"Weights" -> params["arg:pyramid_residual_" <> ToString[i] <> ".residual_denseblock_" <> ToString[j] <> ".comp.conv1.weight"],
		"Biases" -> None, "PaddingSize" -> 0, "Stride" -> 1
	],
	ThreadingLayer[#1 + 0.2#2&]
}, {
	NetPort["Input"] -> 1 -> 2,
	{NetPort["Input"], 2} -> 3
}]
getBlock3[i_, p_] := NetGraph[{
	NetChain@Table[getBlock2[1, n], {n, 1, p}],
	PaddingLayer[{{0, 0}, {1, 1}, {1, 1}}, "Padding" -> "Reflected"],
	ConvolutionLayer[
		"Weights" -> params["arg:pyramid_residual_" <> ToString[i] <> ".final_conv.final_conv.conv.1.weight"],
		"Biases" -> params["arg:pyramid_residual_" <> ToString[i] <> ".final_conv.final_conv.conv.1.bias"],
		"PaddingSize" -> 0, "Stride" -> 1
	],
	ThreadingLayer[Plus]
}, {
	NetPort["Input"] -> 1 -> 2 -> 3,
	{NetPort["Input"], 3} -> 4
}]
getBlock4[i_] := NetChain@{
	PaddingLayer[{{0, 0}, {1, 1}, {1, 1}}, "Padding" -> "Reflected"],
	ConvolutionLayer[
		"Weights" -> params["arg:pyramid_residual_" <> ToString[i] <> "_residual_upsampler.m.0.conv.1.weight"],
		"Biases" -> params["arg:pyramid_residual_" <> ToString[i] <> "_residual_upsampler.m.0.conv.1.bias"],
		"PaddingSize" -> 0, "Stride" -> 1
	]}


resNet = NetChain[{
	ConvolutionLayer[
		"Weights" -> params["arg:sub_mean.weight"],
		"Biases" -> params["arg:sub_mean.bias"],
		"PaddingSize" -> 0, "Stride" -> 1
	],
	PaddingLayer[{{0, 0}, {1, 1}, {1, 1}}, "Padding" -> "Reflected"],
	ConvolutionLayer[
		"Weights" -> params["arg:init_conv_1.conv.1.weight"],
		"Biases" -> params["arg:init_conv_1.conv.1.bias"],
		"PaddingSize" -> 0, "Stride" -> 1
	],
	getBlock3[1, 9],
	getBlock4[1],
	PixelShuffleLayer[2],
	ElementwiseLayer["ReLU"],
	PaddingLayer[{{0, 0}, {1, 1}, {1, 1}}, "Padding" -> "Reflected"],
	ConvolutionLayer[
		"Weights" -> params["arg:reconst_1.final_conv.conv.1.weight"],
		"Biases" -> params["arg:reconst_1.final_conv.conv.1.bias"],
		"PaddingSize" -> 0, "Stride" -> 1
	]
}
];


(* ::Subchapter:: *)
(*Main*)


mainNet = NetGraph[{
	resNet,
(*here should use bicubic in fact*)
	ResizeLayer[{Scaled[2], Scaled[2]}, "Resampling" -> "Linear"],
	ThreadingLayer[#1 / 255 + #2&]
}, {
	NetPort["Input"] -> {1, 2} -> 3
},
	"Input" -> encoder,
	"Output" -> decoder
]


(* ::Subchapter:: *)
(*Export Model*)


Export["ProSR2x trained on Flickr2K.WXF", mainNet]
