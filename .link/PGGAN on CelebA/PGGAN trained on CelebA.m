(* ::Package:: *)

SetDirectory@NotebookDirectory[];
<< NeuralNetworks`
<< MXNetLink`
<< DeepMath`
DateString[]


(* ::Subitem:: *)
(*Thu 25 Oct 2018 18:16:43*)


(* ::Subchapter:: *)
(*Import Weights*)


params = Import@"karras2018iclr-celebahq-1024x1024.wxf";


(* ::Subchapter:: *)
(*Encoder & Decoder*)


encoder = ""
decoder = NetDecoder["Image"]


(* ::Subchapter:: *)
(*Pre-defined Structure*)


leakyReLU[alpha_] := ElementwiseLayer[Ramp[#] - alpha * Ramp[-#]&]
$NCHW = TransposeLayer[{1<->4, 2<->3, 3<->4}];
getWeight[name_String] := Block[
	{raw = Normal@params[name]},
	$NCHW[raw] / Dimensions[raw][[3]]
];
getCN[name_, p_, s_] := ConvolutionLayer[
	"Weights" -> $NCHW[Normal@params[name <> "/weight"]] / Sqrt@s,
	"Biases" -> params[name <> "/bias"],
	"PaddingSize" -> p, "Stride" -> 1
];
getOut[res_, wScale_] := Block[
	{i, weight, bias},
	i = ToString[10 - Log2[res]];
	weight = params["G_paper_1/ToRGB_lod" <> i <> "/weight"];
	bias = params["G_paper_1/ToRGB_lod" <> i <> "/bias"];
	ConvolutionLayer[
		"Weights" -> $NCHW[Normal@weight] / Sqrt[4wScale],
		"Biases" -> (Normal@bias + 1) / 2
	]
];
getBlock[i_, s1_, s2_] := NetChain[{
	ResizeLayer[Scaled /@ {2, 2}, "Resampling" -> "Nearest"],
	getCN[StringRiffle[{"G_paper_1/", i, "x", i, "/Conv0"}, ""], 1, s1],
	leakyReLU[0.2],
	PixelNormalizationLayer[],
	getCN[StringRiffle[{"G_paper_1/", i, "x", i, "/Conv1"}, ""], 1, s2],
	leakyReLU[0.2],
	PixelNormalizationLayer[]
}];


$part1 = NetChain@{
	ReshapeLayer[{512, 4, 4}],
	leakyReLU[0.2],
	PixelNormalizationLayer[],
	getCN["G_paper_1/4x4/Conv", 1, 1],
	leakyReLU[0.2],
	PixelNormalizationLayer[]
};


(* ::Subchapter:: *)
(*Main*)


mainNet = NetChain[{
(*ReshapeLayer[{512,1,1}],*)
	LinearLayer[8192,
		"Weights" -> Transpose[weight / 64],
	(*there's no broadcast in Mathematica*)
		"Biases" -> Flatten@TransposeLayer[{1<->2}][ConstantArray[bias, 16]]
	],
	$part1,
(*I can't understand how to calc wScale*)
	getBlock[8, 2304, 2304],
	getBlock[16, 2304, 2304],
	getBlock[32, 2304, 2304],
	getBlock[64, 2304, 1152],
	getBlock[128, 1152, 576],
	getBlock[256, 576, 288],
	getBlock[512, 288, 144],
	getBlock[1024, 144, 72],
	getOut[1024, 16]
(*ResizeLayer[Scaled/@{1/3,1/3}]*)
},
	"Input" -> 512,
	"Output" -> "Image"
]


(* ::Subchapter:: *)
(*Export Model*)


Export["PGGAN trained on CelebA.WXF", mainNet]
