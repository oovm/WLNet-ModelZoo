(* ::Package:: *)

SetDirectory@NotebookDirectory[];
<< NeuralNetworks`
<< MXNetLink`
<< DeepMath`
DateString[]


(* ::Subitem:: *)
(*Sat 22 Dec 2018 21:22:36*)


(* ::Subchapter:: *)
(*Import Weights*)


params = Import["illust2vec_ver200.caffemodel.wxf"];


(* ::Subchapter:: *)
(*Encoder & Decoder*)


meanImage = Image[Normal[Import@"image_mean.npy.wxf"] / 255, Interleaving -> False];
meanChannel = {
	0.6461231078823529`,
	0.6567790045882352`,
	0.7103466105490196`
};
tags = Import["tag_list.json"];


(* ::Subchapter::Closed:: *)
(*Pre-defined Structure*)


ReLU = ElementwiseLayer["ReLU"];
Pooling = PoolingLayer[{2, 2}, "Stride" -> 2, "Function" -> Max];
getCN[name_String, p_ : 1, s_ : 1] := ConvolutionLayer[
	"Weights" -> params[name <> "_1"],
	"Biases" -> params[name <> "_2"],
	"PaddingSize" -> p, "Stride" -> s
];
getDN[name_String, c_] := LinearLayer[c,
	"Weights" -> params[name <> "_1"],
	"Biases" -> params[name <> "_2"]
];
getBlock[i_, j_] := NetChain@{
	getCN["conv" <> ToString[i] <> "_" <> ToString@j, 1, 1],
	ReLU
};
getBlock2[a_, b_] := NetChain[
	{ PartLayer[a ;; b], ElementwiseLayer[Clip]},
	"Input" -> 1539,
	"Output" -> NetDecoder[{"Class", Capitalize@tags[[a ;; b]]}]
];


(* ::Subchapter:: *)
(*Main*)


input = NetEncoder[{"Image", 224, "MeanImage" -> meanImage, "VarianceImage" -> 1 / 255}];
extractor = NetChain[{
	Table[getBlock[1, j], {j, 1}],
	Pooling,
	Table[getBlock[2, j], {j, 1}],
	Pooling,
	Table[getBlock[3, j], {j, 2}],
	Pooling,
	Table[getBlock[4, j], {j, 2}],
	Pooling,
	Table[getBlock[5, j], {j, 2}],
	Pooling,
	Table[getBlock[6, j], {j, 3}]
},
	"Input" -> input
];
classifier = {
	DropoutLayer[0.5],
	getDN["encode1", 4096],
	LogisticSigmoid,
	getDN["encode2", 1539]
};
general = getBlock2[1, 512];
character = getBlock2[513, 1024];
copyright = getBlock2[1025, 1536];
rating = getBlock2[1537, 1539];

mainNet = NetGraph[{
	"Extractor" -> extractor,
	"Classifier" -> classifier,
	"General" -> general,
	"Character" -> character,
	"Copyright" -> copyright,
	"Rating" -> rating
},
	{
		"Extractor" -> "Classifier" -> {
			"General" -> NetPort["General"],
			"Character" -> NetPort["Character"],
			"Copyright" -> NetPort["Copyright"],
			"Rating" -> NetPort["Rating"]
		}
	}
]


(* ::Subchapter:: *)
(*Export Model*)


Export["Illustration2Vec Mega trained on Danbooru.WLNet", mainNet]
