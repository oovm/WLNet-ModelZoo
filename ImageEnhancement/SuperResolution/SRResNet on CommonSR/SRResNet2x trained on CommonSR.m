(* ::Package:: *)

SetDirectory@NotebookDirectory[];
Needs["MXNetLink`"]
Needs["NeuralNetworks`"]
DateString[]


(* ::Subitem:: *)
(*Sat 27 Oct 2018 15:08:10*)


(* ::Subchapter:: *)
(*Import Weights*)


params = NDArrayImport["SRResNet_bicx2_in3nf64nb16-0000.params"];


(* ::Subchapter:: *)
(*Encoder & Decoder*)


encoder = NetEncoder[{"Image", {640, 360}}]
decoder = NetDecoder["Image"]


(* ::Subchapter::Closed:: *)
(*Pre-defined Structure*)


getCN[i_, s_, p_] := ConvolutionLayer[
	"Weights" -> params["arg:learned_" <> ToString[i]],
	"Biases" -> params["arg:learned_" <> ToString[i + 1]],
	"PaddingSize" -> p, "Stride" -> s
]
getBlock[i_] := NetGraph[{
	getCN[i, 1, 1],
	ElementwiseLayer["ReLU"],
	getCN[i + 2, 1, 1],
	ThreadingLayer[Plus]
}, {
	NetPort["Input"] -> 1 -> 2 -> 3,
	{NetPort["Input"], 3} -> 4
}];
$head = getCN[0, 1, 1];
$body = NetGraph[{
	NetChain@Table[getBlock[i], {i, 2, 64, 4}],
	getCN[66, 1, 1],
	ThreadingLayer[Plus]
}, {
	NetPort["Input"] -> 1 -> 2 ,
	{NetPort["Input"], 2} -> 3
}];
$tail = NetChain@{
	getCN[68, 1, 1],
	PixelShuffleLayer[2],
	ElementwiseLayer["ReLU"],
	getCN[70, 1, 1],
	ElementwiseLayer["ReLU"],
	getCN[72, 1, 1]
};


(* ::Subchapter:: *)
(*Main*)


mainNet = NetChain[{$head, $body, $tail}] // NetFlatten;
mainNet = NetReplacePart[mainNet, {"Input" -> encoder, "Output" -> decoder}]


(* ::Subchapter:: *)
(*Export Model*)


Export["SRResNet2x trained on CommonSR.WXF", mainNet]
