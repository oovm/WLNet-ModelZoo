(* ::Package:: *)

SetDirectory@NotebookDirectory[];
<< NeuralNetworks`
<< MXNetLink`
<< DeepMath`
DateString[]


(* ::Subitem:: *)
(*Tue 25 Nov 2018 21:16:33*)


(* ::Subchapter:: *)
(*Import Weights*)


params = NDArrayImport["CBDNet_JPEG-0000.params"];


(* ::Subchapter:: *)
(*Encoder & Decoder*)


encoder = NetEncoder[{"Image", {640, 360}}]
decoder = NetDecoder["Image"]


(* ::Subchapter:: *)
(*Pre-defined Structure*)


getCN[i_, p_, s_] := ConvolutionLayer[
	"Weights" -> params["arg:convolution" <> ToString@i <> "_weight"],
	"Biases" -> params["arg:convolution" <> ToString@i <> "_bias"],
	"PaddingSize" -> p, "Stride" -> s
];
getDN[i_, p_, s_] := DeconvolutionLayer[
	"Weights" -> params["arg:deconvolution" <> ToString@i <> "_weight"],
	"Biases" -> None, "PaddingSize" -> p, "Stride" -> s
];


sub = Flatten@Table[{getCN[i, 1, 1], ElementwiseLayer["ReLU"]}, {i, 0, 4}];
estimate = NetMerge[NetChain@sub, Join, Expand -> All];
line1 = NetFlatten@NetChain@Table[{getCN[i, 1, 1], ElementwiseLayer["ReLU"]}, {i, 5, 8}];
line2 = NetChain@Flatten@{
	{getCN[9, 0, 2], ElementwiseLayer["ReLU"]},
	{getCN[10, 0, 1], ElementwiseLayer["ReLU"]},
	Table[{getCN[i, 1, 1], ElementwiseLayer["ReLU"]}, {i, 11, 13}]
};
line3 = NetChain@Flatten@{
	{getCN[14, 0, 2], ElementwiseLayer["ReLU"]},
	{getCN[15, 0, 1], ElementwiseLayer["ReLU"]},
	Table[{getCN[i, 1, 1], ElementwiseLayer["ReLU"]}, {i, 16, 22}],
	getDN[0, 0, 2]
};
line4 = NetChain@Flatten@{
	Table[{getCN[i, 1, 1], ElementwiseLayer["ReLU"]}, {i, 23, 25}],
	getDN[1, 0, 2]
};
line5 = NetChain@Flatten@Table[{getCN[i, 1, 1], ElementwiseLayer["ReLU"]}, {i, 26, 28}];


(* ::Subchapter:: *)
(*Main*)


mainNet = NetGraph[{
	"Estimate" -> sub,
	"Join" -> CatenateLayer[],
	"EncoderA" -> line1,
	"EncoderB" -> line2,
	"Main" -> line3,
	"DecoderB" -> line4,
	"DecoderA" -> line5,
	"Add_1" -> ThreadingLayer[Plus],
	"Add_2" -> ThreadingLayer[Plus],
	"Add_3" -> ThreadingLayer[Plus]
}, {
	NetPort["Input"] -> "Estimate",
	{NetPort["Input"], "Estimate"} -> "Join" -> "EncoderA" -> "EncoderB" -> "Main",
	{"EncoderB", "Main"} -> "Add_1" -> "DecoderB",
	{"EncoderA", "DecoderB"} -> "Add_2" -> "DecoderA",
	{NetPort["Input"], "DecoderA"} -> "Add_3" -> NetPort["Output"]
}]


testNet = NetReplacePart[
	mainNet, {
		"Input" -> encoder,
		"Output" -> decoder
	}
]


(* ::Subchapter:: *)
(*Export Model*)


Export["CBDNet-JPEG trained on CommonNoise.WXF", mainNet]
