(* ::Package:: *)

SetDirectory@NotebookDirectory[];
<< NeuralNetworks`
<< MXNetLink`
<< DeepMath`
DateString[]


(* ::Subitem:: *)
(*Sat 22 Dec 2018 20:28:50*)


(* ::Subchapter:: *)
(*Import Weights*)


params = Import["illust2vec_tag_ver200.caffemodel.wxf"];


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
getBlock[i_, j_] := NetChain@{
	getCN["conv" <> ToString[i] <> "_" <> ToString@j, 1, 1],
	ReLU
};


(* ::Subchapter:: *)
(*Main*)


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
	"Input" -> NetEncoder[{"Image", 224, "MeanImage" -> meanImage}]
];
classifier = {
	DropoutLayer[0.5],
	getBlock[6, 4],
	AggregationLayer@Mean
};
mainNet = NetChain[{
	"Extractor" -> extractor,
	"Classifier" -> classifier
}

]


(* ::Subchapter:: *)
(*Export Model*)


Export["Illustration2Vec trained on Danbooru.WXF", mainNet]
