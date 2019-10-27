(* ::Package:: *)

SetDirectory@NotebookDirectory[];


(* ::Subchapter:: *)
(*Import Weights*)


<<NeuralNetworks`
NeuralNetworks`PixelNormalizationLayer;
file = "PixelNorm.m";
def = NeuralNetworks`Private`ReadDefinitionFile[file, "System`"];
NeuralNetworks`DefineLayer["PixelNorm", def];


params = Import@"PGGAN-128 trained on Anime.WXF";


(* ::Subchapter:: *)
(*Pre-defined Structure*)


leakyReLU[alpha_] := ElementwiseLayer[Ramp[#] - alpha * Ramp[-#]&];
$NCHW = TransposeLayer[{1<->4, 2<->3, 3<->4}];
trans = NetGraph[
	<|
		"pad" -> PaddingLayer[{{1, 1}, {1, 1}, {0, 0}, {0, 0}}],
		"t4" -> PartLayer[{2 ;; 5, 2 ;; 5}],
		"t3" -> PartLayer[{1 ;; 4, 2 ;; 5}],
		"t2" -> PartLayer[{2 ;; 5, 1 ;; 4}],
		"t1" -> PartLayer[{1 ;; 4, 1 ;; 4}],
		"add" -> ThreadingLayer[Plus]
	|>,
	{
		NetPort["Input"] -> "pad",
		"pad" -> {"t1", "t2", "t3", "t4"} -> "add",
		"add" ->  NetPort["Output"]
	}
];
getCN[name_, s_] := ConvolutionLayer[
	"Weights" -> $NCHW[Normal@params[name <> "/weight"]] / Sqrt@s,
	"Biases" -> params[name <> "/bias"],
	 "Stride" -> 1,"PaddingSize" -> 1
];
getDN[name_, s_] := DeconvolutionLayer[
	"Weights" -> $NCHW[trans@Normal@params[name <> "/weight"]] / Sqrt@s,
	"Biases" -> params[name <> "/bias"],
	"Stride" -> 2, "PaddingSize" -> 1
];
getBlock[i_, s1_, s2_] := NetChain[{
	getDN[StringRiffle[{"Gs/", i, "x", i, "/Conv0_up"}, ""], s1],
	leakyReLU[0.2],
	PixelNormalizationLayer[],
	getCN[StringRiffle[{"Gs/", i, "x", i, "/Conv1"}, ""], s2],
	leakyReLU[0.2],
	PixelNormalizationLayer[]
}];


(* ::Subchapter:: *)
(*Main*)


$part1 = NetChain@{
	ReshapeLayer[{512, 4, 4}],
	leakyReLU[0.2],
	PixelNormalizationLayer[],
	getCN["Gs/4x4/Conv", 2304],
	leakyReLU[0.2],
	PixelNormalizationLayer[]
};


mainNet = NetChain[{
	PixelNormalizationLayer[],
	LinearLayer[8192,
		"Weights" -> Transpose[Normal@params["Gs/4x4/Dense/weight"] /64],
		"Biases" -> Flatten@TransposeLayer[{1<->2}][ConstantArray[Normal@params["Gs/4x4/Dense/bias"], 16]]
	],
	$part1,
	getBlock[8, 2304, 2304],
	getBlock[16, 2304, 2304],
	getBlock[32, 2304, 2304],
	getBlock[64, 2304, 1152],
	getBlock[128, 1152, 576],
	(*getBlock[256, 576, 288],*)
	(*getBlock[512, 288, 144],*)
	ConvolutionLayer[
		"Weights" -> $NCHW[Normal@params["Gs/ToRGB_lod2/weight"]] / Sqrt[128],
		"Biases" -> Normal@params["Gs/ToRGB_lod2/bias"],
		"PaddingSize" -> 0, "Stride" -> 1
	],
	ElementwiseLayer[(Tanh[#*1.414] + 1) /2 &]
},
	"Input" -> 512,
	"Output" -> "Image"
]


(* ::Subchapter:: *)
(*Export Model*)


Export["PGGAN-128 trained on Anime.WLNet", mainNet]
