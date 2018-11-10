(* ::Package:: *)

SetDirectory@NotebookDirectory[];
<< NeuralNetworks`
<< MXNetLink`
<< DeepMath`
DateString[]


(* ::Subitem:: *)
(*Sat 10 Nov 2018 14:41:52*)


(* ::Subchapter:: *)
(*Import Weights*)


params = Import@"karras2018iclr-celebahq-1024x1024.wxf";


(* ::Subchapter:: *)
(*Encoder & Decoder*)


encoder = ""
decoder = NetDecoder["Image"]


(* ::Subchapter::Closed:: *)
(*Pre-defined Structure*)


leakyReLU[alpha_] := ElementwiseLayer[Ramp[#] - alpha * Ramp[-#]&]
$NCHW = TransposeLayer[{1<->4, 2<->3, 3<->4}];
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
}]


$pre = LinearLayer[8192,
	"Weights" -> Transpose[Normal@params["G_paper_1/4x4/Dense/weight"] / 64],
(*there's no broadcast in Mathematica*)
	"Biases" -> Flatten@TransposeLayer[{1<->2}][ConstantArray[Normal@params["G_paper_1/4x4/Dense/bias"], 16]]
];
$part1 = NetChain@{
	ReshapeLayer[{512, 4, 4}],
	leakyReLU[0.2],
	PixelNormalizationLayer[],
	getCN["G_paper_1/4x4/Conv", 1, 1],
	leakyReLU[0.2],
	PixelNormalizationLayer[]
}


(* ::Subchapter:: *)
(*Main*)


mainNet = NetChain[{
(*ReshapeLayer[{512,1,1}],*)
	$pre,
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


Table[StringJoin[{ToString@i, "*", ToString@i, "_Gen"}]
	-> StringJoin[{ToString[2i], "*", ToString[2i], "_Gen"}]
	-> StringJoin[{ToString[2i], "*", ToString[2i], "_Out"}]
	-> NetPort[StringJoin[{ToString[2i], "*", ToString[2i], ""}]],
	{i, PowerRange[4, 2^9, 2]}
];

fullNet = NetGraph[
	{
		"4*4_Pre" -> $pre,
		"4*4_Gen" -> $part1,
		"4*4_Out" -> getOut[4, 512],
		"8*8_Gen" -> getBlock[8, 2304, 2304],
		"8*8_Out" -> getOut[8, 512],
		"16*16_Gen" -> getBlock[16, 2304, 2304],
		"16*16_Out" -> getOut[16, 512],
		"32*32_Gen" -> getBlock[32, 2304, 2304],
		"32*32_Out" -> getOut[32, 512],
		"64*64_Gen" -> getBlock[64, 2304, 1152],
		"64*64_Out" -> getOut[64, 256],
		"128*128_Gen" -> getBlock[128, 1152, 576],
		"128*128_Out" -> getOut[128, 128],
		"256*256_Gen" -> getBlock[256, 576, 288],
		"256*256_Out" -> getOut[256, 64],
		"512*512_Gen" -> getBlock[512, 288, 144],
		"512*512_Out" -> getOut[512, 32],
		"1024*1024_Gen" -> getBlock[1024, 144, 72],
		"1024*1024_Out" -> getOut[1024, 16]
	},
	{
		NetPort["Input"] -> "4*4_Pre",
		"4*4_Pre" -> "4*4_Gen" -> "4*4_Out" -> NetPort["4*4"],
		"4*4_Gen" -> "8*8_Gen" -> "8*8_Out" -> NetPort["8*8"],
		"8*8_Gen" -> "16*16_Gen" -> "16*16_Out" -> NetPort["16*16"],
		"16*16_Gen" -> "32*32_Gen" -> "32*32_Out" -> NetPort["32*32"],
		"32*32_Gen" -> "64*64_Gen" -> "64*64_Out" -> NetPort["64*64"],
		"64*64_Gen" -> "128*128_Gen" -> "128*128_Out" -> NetPort["128*128"],
		"128*128_Gen" -> "256*256_Gen" -> "256*256_Out" -> NetPort["256*256"],
		"256*256_Gen" -> "512*512_Gen" -> "512*512_Out" -> NetPort["512*512"],
		"512*512_Gen" -> "1024*1024_Gen" -> "1024*1024_Out" -> NetPort["1024*1024"]
	},
	"Input" -> 512,
	"4*4" -> "Image",
	"8*8" -> "Image",
	"16*16" -> "Image",
	"32*32" -> "Image",
	"64*64" -> "Image",
	"128*128" -> "Image",
	"256*256" -> "Image",
	"512*512" -> "Image",
	"1024*1024" -> "Image"
]


(* ::Subchapter:: *)
(*Export Model*)


Export["PGGAN trained on CelebA.WXF", mainNet]
