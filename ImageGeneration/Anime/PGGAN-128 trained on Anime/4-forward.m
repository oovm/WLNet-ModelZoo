(* ::Package:: *)

SetDirectory@NotebookDirectory[];


<< NeuralNetworks`
NeuralNetworks`PixelNormalizationLayer;
file = "PixelNorm.m";
def = NeuralNetworks`Private`ReadDefinitionFile[file, "NeuralNetworks`"];
NeuralNetworks`DefineLayer["PixelNorm", def];


mainNet = Import@"PGGAN-128 trained on Anime.WLNet";


SeedRandom[42];
inBatch = RandomVariate[NormalDistribution[], {300, 512}];
outBatch = mainNet[inBatch, TargetDevice -> "GPU"];


MapIndexed[First@#2 -> #1&, outBatch];
pick = {
	1, 2, 4, 5, 6, 10, 14, 15, 17, 18, 23, 24, 27, 32, 33, 34, 37, 38, 39, 40, 44, 45, 46, 47, 49, 53, 54, 57, 59, 62, 63, 69, 72, 73, 75, 76, 79, 82, 84, 87, 89, 90, 91, 92, 96, 97, 98, 100,
	104, 109, 110, 113, 114, 117, 118, 119, 122, 123, 126, 127, 129, 132, 133, 136, 140, 141, 143, 144, 145, 146, 147, 148, 153, 159, 161, 163, 164, 166, 167, 168, 171, 177, 180, 181, 185, 186, 188, 189, 190, 193, 197, 200,
	202, 203, 204, 205, 206, 207, 214, 215, 216, 217, 226, 230, 231, 232, 233, 237, 245, 246, 248, 250, 253, 257, 258, 259, 261, 263, 264, 266, 269, 277, 280, 281, 282, 283, 284, 285, 286, 288, 289, 290, 292, 294, 295, 296, 297, 299, 300
};
Export["preview.png", ImageCollage[RandomSample[outBatch[[pick]], 64]]]
