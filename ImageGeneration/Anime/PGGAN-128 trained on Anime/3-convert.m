(* ::Package:: *)

(* ::Subchapter:: *)
(*Import Weights*)


SetDirectory@NotebookDirectory[];
<< NeuralNetworks`
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
		"add" -> NetPort["Output"]
	}
];
getCN[name_, s_] := ConvolutionLayer[
	"Weights" -> $NCHW[Normal@params[name <> "/weight"]] / Sqrt@s,
	"Biases" -> params[name <> "/bias"],
	"Stride" -> 1, "PaddingSize" -> 1
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
		"Weights" -> Transpose[Normal@params["Gs/4x4/Dense/weight"] / 64],
		"Biases" -> Flatten@TransposeLayer[{1<->2}][ConstantArray[Normal@params["Gs/4x4/Dense/bias"], 16]]
	],
	$part1,
	getBlock[8, 2304, 2304],
	getBlock[16, 2304, 2304],
	getBlock[32, 2304, 2304],
	getBlock[64, 2304, 1152],
	getBlock[128, 1152, 576],
	ConvolutionLayer[
		"Weights" -> Sqrt[8] * $NCHW[Normal@params["Gs/ToRGB_lod2/weight"]] / Sqrt[128],
		"Biases" -> Sqrt[8] * Normal@params["Gs/ToRGB_lod2/bias"],
		"PaddingSize" -> 0, "Stride" -> 1
	],
	LogisticSigmoid
},
	"Input" -> 512,
	"Output" -> "Image"
]


(* ::Subchapter:: *)
(*Export Model*)


Export["PGGAN-128 trained on Anime.MAT", mainNet, "WXF", PerformanceGoal -> "Speed"]


(* ::Subchapter:: *)
(*Testing Model*)


SetDirectory@NotebookDirectory[];
mainNet = Import["PGGAN-128 trained on Anime.MAT", "WXF"];


SeedRandom[42];
inBatch = RandomVariate[NormalDistribution[], {300, 512}];
outBatch = mainNet[inBatch, TargetDevice -> "GPU"];
MapIndexed[First@#2 -> #1&, outBatch];


pick = {
	1, 2, 4, 5, 6, 10, 14, 15, 17, 18, 23, 24, 27, 32, 33, 34, 37, 38, 39, 40, 44, 45, 46, 47, 49, 53, 54, 57, 59, 62, 63, 69, 72, 73, 75, 76, 79, 82, 84, 87, 89, 90, 91, 92, 96, 97, 98, 100,
	104, 109, 110, 113, 114, 117, 118, 119, 122, 123, 126, 127, 129, 132, 133, 136, 140, 141, 143, 144, 145, 146, 147, 148, 153, 159, 161, 163, 164, 166, 167, 168, 171, 177, 180, 181, 185, 186, 188, 189, 190, 193, 197, 200,
	202, 203, 204, 205, 206, 207, 214, 215, 216, 217, 226, 230, 231, 232, 233, 237, 245, 246, 248, 250, 253, 257, 258, 259, 261, 263, 264, 266, 269, 277, 280, 281, 282, 283, 284, 285, 286, 288, 289, 290, 292, 294, 295, 296, 297, 299, 300
};
Export["preview-2.jpg", ImageCollage[RandomSample[outBatch[[pick]], 64]]]
Export["preview.jpg", ImageCollage[RandomSample[outBatch[[pick]], 25]]]
