(* ::Package:: *)

SetDirectory@NotebookDirectory[];
<< NeuralNetworks`
<< MXNetLink`
<< DeepMath`
DateString[]


(* ::Subitem:: *)
(*Fri 16 Nov 2018 00:26:35*)


(* ::Subchapter:: *)
(*Import Weights*)


raw = Import["color.wxf"];
params[name_String] := Block[
	{$NCHW, prefix, input},
	$NCHW = TransposeLayer[{1<->4, 2<->3, 3<->4}];
	prefix = "g_net/";
	input = Normal@raw[prefix <> name];
	Switch[
		Length@Dimensions@input,
		1, RawArray["Real32", input],
		4, RawArray["Real32", $NCHW[input]],
		_, RawArray["Real32", input]
	]
]


(* ::Subchapter:: *)
(*Encoder & Decoder*)


(* ::Subchapter:: *)
(*Pre-defined Structure*)


getCN[name_String, p_, s_, ops___] := ConvolutionLayer[
	"Weights" -> params[name <> "/weights"],
	"Biases" -> params[name <> "/biases"],
	"PaddingSize" -> p, "Stride" -> s, ops
];
getDN[name_String, p_, s_, ops___] := DeconvolutionLayer[
	"Weights" -> params[name <> "/weights"],
	"Biases" -> params[name <> "/biases"],
	"PaddingSize" -> p, "Stride" -> s, ops
];
getEncBlock[i_, j_] := Block[
	{path},
	path = NetChain@{
		getCN["enc" <> ToString[i] <> "_" <> ToString[j] <> "/conv1", 2, 1],
		ElementwiseLayer["ReLU"],
		getCN["enc" <> ToString[i] <> "_" <> ToString[j] <> "/conv2", 2, 1]
	};
	NetMerge[path, Expand -> All]
];
getEnc[i_, stride_ : 1] := Flatten[{
	getCN["enc" <> ToString[i] <> "_1", 2, stride],
	ElementwiseLayer["ReLU"],
	Table[getEncBlock[i, j], {j, 2, 4}]
}] // NetChain;
getDecBlock[i_, j_] := Block[
	{path},
	path = NetChain@{
		getCN["dec" <> ToString[i] <> "_" <> ToString[j] <> "/conv1", 2, 1],
		ElementwiseLayer["ReLU"],
		getCN["dec" <> ToString[i] <> "_" <> ToString[j] <> "/conv2", 2, 1]
	};
	NetMerge[path, Expand -> All]
];
getDec[i_] := Flatten[{
	Table[getDecBlock[i, j], {j, 3, 1, -1}],
	getDN["dec" <> ToString[i - 1] <> "_4", 1, 2],
	ElementwiseLayer["ReLU"]
}] // NetChain;


(* ::Subchapter:: *)
(*Main*)


mainNet = NetGraph[{
	"Double" -> CatenateLayer[],
	"Enc_1" -> getEnc[1, 1],
	"Enc_2" -> getEnc[2, 2],
	"Enc_3" -> getEnc[3, 2],
	"Dec_3" -> getDec[3],
	"Add_3" -> ThreadingLayer[Plus],
	"Dec_2" -> getDec[2],
	"Add_2" -> ThreadingLayer[Plus],
	"Dec_1" -> Flatten@{
		Table[getDecBlock[1, j], {j, 3, 1, -1}],
		getCN["dec1_0", 2, 1]
	}
}, {
	{NetPort["Input"], NetPort["Input"]} -> "Double",
	"Double" -> "Enc_1" -> "Enc_2" -> "Enc_3" -> "Dec_3",
	{"Enc_2", "Dec_3"} -> "Add_3" -> "Dec_2",
	{"Enc_1", "Dec_2"} -> "Add_2" -> "Dec_1"
}]


(* ::Subchapter:: *)
(*Test*)


evalNet[img_] := NetChain[
	{mainNet},
	"Input" -> NetEncoder[{"Image", ImageDimensions@img}],
	"Output" -> NetDecoder["Image"]
][img, TargetDevice -> "GPU"]


(* ::Subchapter:: *)
(*Export Model*)


Export["SRN Deblur trained on GOPRO.WXF", mainNet]
