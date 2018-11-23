(* ::Package:: *)

SetDirectory@NotebookDirectory[];
Needs["MXNetLink`"]
Needs["NeuralNetworks`"]
DateString[]


(* ::Subitem:: *)
(*Sat 27 Oct 2018 18:15:08*)


(* ::Subchapter:: *)
(*Import Weights*)


params = NDArrayImport["SRGAN_8x-0000.params"];


(* ::Subchapter:: *)
(*Encoder & Decoder*)


encoder = NetEncoder[{"Image", {640, 360}}]
decoder = NetDecoder["Image"]


(* ::Subchapter:: *)
(*Pre-defined Structure*)


getCN[i_, s_, p_] := ConvolutionLayer[
	"Weights" -> params["arg:learned_" <> ToString[i]],
	"Biases" -> params["arg:learned_" <> ToString[i + 1]],
	"PaddingSize" -> p, "Stride" -> s
]
getBN[i_] := BatchNormalizationLayer[
	"Epsilon" -> 1*^-5,
	"Gamma" -> params["arg:learned_" <> ToString[i]],
	"Beta" -> params["arg:learned_" <> ToString[i + 1]],
	"MovingMean" -> params["aux:learned_" <> ToString[i + 2]],
	"MovingVariance" -> params["aux:learned_" <> ToString[i + 3]]
]

(*Notice that there is no single prelu in WLNet*)
getPrelu[i_] := ParametricRampLayer[
	"Slope" -> Flatten@ConstantArray[Normal@params["arg:learned_" <> ToString[i]], 64]
]

getBlock[i_] := NetGraph[{
	getCN[i, 1, 1],
	getBN[i + 2],
	getPrelu[i + 6],
	getCN[i + 7, 1, 1],
	getBN[i + 9],
	ThreadingLayer[Plus]
}, {
	NetPort["Input"] -> 1 -> 2 -> 3 -> 4 -> 5,
	{NetPort["Input"], 5} -> 6
}];


$head = NetChain@{
	getCN[0, 1, 4],
	getPrelu[2]
};
$body = NetGraph[{
	NetChain@Table[getBlock[i], {i, 3, 55, 13} ],
	getCN[68, 1, 1],
	getPrelu[70],
	ThreadingLayer[Plus]
}, {
	NetPort["Input"] -> 1 -> 2 -> 3 ,
	{NetPort["Input"], 3} -> 4
}];
$tail = NetChain@{
	getCN[71, 1, 1],
	PixelShuffleLayer[2],
	getPrelu[73],
	getCN[74, 1, 1],
	PixelShuffleLayer[2],
	getPrelu[76],
	getCN[77, 1, 1],
	PixelShuffleLayer[2],
	getPrelu[79],
	getCN[80, 1, 4],
(*LogisticSigmoid[2x]==(Tanh[x]+1)/2*)
	ElementwiseLayer[(Tanh[#] + 1) / 2&]
};


(* ::Subchapter:: *)
(*Main*)


mainNet = NetChain[{$head, $body, $tail}] // NetFlatten;
mainNet = NetReplacePart[mainNet, {"Input" -> encoder, "Output" -> decoder}]


(* ::Subchapter:: *)
(*Export Model*)


Export["SRGAN8x trained on VOC.WXF", mainNet]
