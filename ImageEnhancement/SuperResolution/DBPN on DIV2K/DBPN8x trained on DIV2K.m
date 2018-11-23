(* ::Package:: *)

SetDirectory@NotebookDirectory[];
Needs["MXNetLink`"]
Needs["NeuralNetworks`"]
DateString[]


(* ::Subitem:: *)
(*Wed 24 Oct 2018 22:23:27*)


(* ::Subchapter:: *)
(*Import Weights*)


params = NDArrayImport@"DBPN_8x-0000.params";


(* ::Subchapter:: *)
(*Encoder & Decoder*)


encoder = NetEncoder[{"Image", {640, 360}}]
decoder = NetDecoder["Image"]


(* ::Subchapter:: *)
(*Pre-defined Structure*)


sJ[s_String, n_Integer] := s <> ToString[n]
(*Down-Projection Unit*)
getC[nn_] := Block[
	{n = ToString[nn]},
	NetGraph[{
		ConvolutionLayer[
			"Weights" -> params["arg:conv1_bp" <> n <> "_weight"],
			"Biases" -> params[StringJoin["arg:conv1_bp" <> n <> "_bias"]],
			"PaddingSize" -> 2, "Stride" -> 8
		],
		ParametricRampLayer["Slope" -> params["arg:relu1_bp" <> n <> "_gamma"]],
		DeconvolutionLayer[
			"Weights" -> params["arg:conv2_bp" <> n <> "_weight"],
			"Biases" -> params[StringJoin["arg:conv2_bp" <> n <> "_bias"]],
			"PaddingSize" -> 2, "Stride" -> 8
		],
		ParametricRampLayer["Slope" -> params["arg:relu2_bp" <> n <> "_gamma"]],
		ThreadingLayer[#1 - #2 &],
		ConvolutionLayer[
			"Weights" -> params["arg:conv3_bp" <> n <> "_weight"],
			"Biases" -> params[StringJoin["arg:conv3_bp" <> n <> "_bias"]],
			"PaddingSize" -> 2, "Stride" -> 8
		],
		ParametricRampLayer["Slope" -> params["arg:relu3_bp" <> n <> "_gamma"]],
		ThreadingLayer[#1 + #2 &]
	}, {
		NetPort["Input"] -> 1 -> 2 -> 8,
		{NetPort["Input"], 4} -> 5,
		2 -> 3 -> 4, 5 -> 6 -> 7 -> 8
	}]
];
getCo[n_] := NetGraph[{
	ConvolutionLayer[
		"Weights" -> params["arg:out_concat_bp" <> ToString[n - 1] <> "_weight"],
		"Biases" -> params[StringJoin["arg:out_concat_bp" <> ToString[n - 1] <> "_bias"]],
		"PaddingSize" -> 0, "Stride" -> 1
	],
	ParametricRampLayer["Slope" -> params["arg:relu1_concat_bp" <> ToString[n - 1] <> "_gamma"]],
	getC[n]
},
	{1 -> 2 -> 3}
] // NetFlatten;

(*Up-Projection Unit*)
getD[nn_] := Block[
	{n = ToString[nn]},
	NetGraph[{
		DeconvolutionLayer[
			"Weights" -> params["arg:conv1_bp" <> n <> "_weight"],
			"Biases" -> params[StringJoin["arg:conv1_bp" <> n <> "_bias"]],
			"PaddingSize" -> 2, "Stride" -> 8
		],
		ParametricRampLayer["Slope" -> params["arg:relu1_bp" <> n <> "_gamma"]],
		ConvolutionLayer[
			"Weights" -> params["arg:conv2_bp" <> n <> "_weight"],
			"Biases" -> params[StringJoin["arg:conv2_bp" <> n <> "_bias"]],
			"PaddingSize" -> 2, "Stride" -> 8
		],
		ParametricRampLayer["Slope" -> params["arg:relu2_bp" <> n <> "_gamma"]],
		ThreadingLayer[#1 - #2 &],
		DeconvolutionLayer[
			"Weights" -> params["arg:conv3_bp" <> n <> "_weight"],
			"Biases" -> params[StringJoin["arg:conv3_bp" <> n <> "_bias"]],
			"PaddingSize" -> 2, "Stride" -> 8
		],
		ParametricRampLayer["Slope" -> params["arg:relu3_bp" <> n <> "_gamma"]],
		ThreadingLayer[#1 + #2 &]
	}, {
		NetPort["Input"] -> 1 -> 2 -> 8,
		{NetPort["Input"], 4} -> 5,
		2 -> 3 -> 4, 5 -> 6 -> 7 -> 8
	}]
];
getDo[n_] := NetGraph[{
	ConvolutionLayer[
		"Weights" -> params["arg:out_concat_bp" <> ToString[n - 1] <> "_weight"],
		"Biases" -> params[StringJoin["arg:out_concat_bp" <> ToString[n - 1] <> "_bias"]],
		"PaddingSize" -> 0, "Stride" -> 1
	],
	ParametricRampLayer["Slope" -> params["arg:relu1_concat_bp" <> ToString[n - 1] <> "_gamma"]],
	getD[n]
},
	{1 -> 2 -> 3}
] // NetFlatten;


(* ::Subchapter:: *)
(*Main*)


$head = NetChain@{
	SequenceReverseLayer[],
	ConvolutionLayer[
		"Weights" -> params["arg:conv0_1_weight"],
		"Biases" -> params[StringJoin["arg:conv0_1_bias"]],
		"PaddingSize" -> 1, "Stride" -> 1
	],
	ParametricRampLayer["Slope" -> params["arg:relu0_1_gamma"]],
	ConvolutionLayer[
		"Weights" -> params["arg:conv1_weight"],
		"Biases" -> params[StringJoin["arg:conv1_bias"]],
		"PaddingSize" -> 0, "Stride" -> 1
	],
	ParametricRampLayer["Slope" -> params["arg:relu1_gamma"]]
};
$tail = ConvolutionLayer[
	"Weights" -> params["arg:conv_final_weight"],
	"Biases" -> params[StringJoin["arg:conv_final_bias"]],
	"PaddingSize" -> 1, "Stride" -> 1
];
nodes = Flatten@{
	"head" -> $head,
	"up_1" -> getD[1],
	"down_1" -> getC[2],
	"up_2" -> getD[3],
	Table[sJ["down_", i] -> getCo[2i], {i, 2, 6}],
	Table[sJ["up_", i] -> getDo[2i - 1], {i, 3, 7}],
	Table[sJ["cate_", i] -> CatenateLayer[], {i, 1, 11}],
	"tail" -> $tail
};
path = Flatten@{
	NetPort["Input"] -> "head" -> "up_1" -> "down_1" -> "up_2",
	{"up_2", "up_1"} -> "cate_1" -> "down_2",
	{"down_2", "down_1"} -> "cate_2" -> "up_3",
	Table[{sJ["up_", i], sJ["cate_", 2i - 5]} -> sJ["cate_", 2i - 3] -> sJ["down_", i], {i, 3, 6}],
	Table[{sJ["down_", i], sJ["cate_", 2i - 4]} -> sJ["cate_", 2i - 2] -> sJ["up_", i + 1], {i, 3, 6}],
	{"up_7", "cate_9"} -> "cate_11" -> "tail"
};


mainNet = NetGraph[nodes, path, "Input" -> encoder, "Output" -> decoder]


(* ::Subchapter:: *)
(*Export Model*)


Export["DBPN8x trained on DIV2K.WXF", mainNet]
