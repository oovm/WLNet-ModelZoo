(* ::Package:: *)

SetDirectory@NotebookDirectory[];
Needs["MXNetLink`"]
Needs["NeuralNetworks`"]
DateString[]


(* ::Subitem:: *)
(*Wed 17 Oct 2018 12:44:18*)


(* ::Subchapter:: *)
(*Import Weights*)


params = NDArrayImport["imagenet_densenet121-0000.params"];


(* ::Subchapter:: *)
(*Encoder & Decoder*)


mShift = {0.485, 0.456, 0.406};
vShift = {0.229, 0.224, 0.225}^2;
encoder = NetEncoder[{"Image", 224, "MeanImage" -> mShift, "VarianceImage" -> vShift}]
decoder = NetExtract[NetModel["ResNet-50 Trained on ImageNet Competition Data"], "Output"]


(* ::Subchapter::Closed:: *)
(*Pre-defined Structure*)


getBN[i_, j_] := BatchNormalizationLayer[
	"Epsilon" -> 1*^-5,
	"Beta" -> params["arg:densenet0_stage" <> i <> "_batchnorm" <> j <> "_beta"],
	"Gamma" -> params["arg:densenet0_stage" <> i <> "_batchnorm" <> j <> "_gamma"],
	"MovingMean" -> params["aux:densenet0_stage" <> i <> "_batchnorm" <> j <> "_running_mean"],
	"MovingVariance" -> params["aux:densenet0_stage" <> i <> "_batchnorm" <> j <> "_running_var"]
];
getBN2[j_] := BatchNormalizationLayer[
	"Epsilon" -> 1*^-5,
	"Beta" -> params["arg:densenet0_batchnorm" <> j <> "_beta"],
	"Gamma" -> params["arg:densenet0_batchnorm" <> j <> "_gamma"],
	"MovingMean" -> params["aux:densenet0_batchnorm" <> j <> "_running_mean"],
	"MovingVariance" -> params["aux:densenet0_batchnorm" <> j <> "_running_var"]
];
getCN[i_, j_, p_ : 1, s_ : 1] := ConvolutionLayer[
	"Weights" -> params["arg:densenet0_stage" <> i <> "_conv" <> j <> "_weight"],
	"Biases" -> None, "PaddingSize" -> p, "Stride" -> s
];
getCN2[j_, p_ : 1, s_ : 1] := ConvolutionLayer[
	"Weights" -> params["arg:densenet0_conv" <> j <> "_weight"],
	"Biases" -> None, "PaddingSize" -> p, "Stride" -> s
];
getBlock[i_, j_] := NetGraph[{
	getBN[ToString[i], ToString[j]], Ramp, getCN[ToString[i], ToString[j], 0, 1],
	getBN[ToString[i], ToString[j + 1]], Ramp, getCN[ToString[i], ToString[j + 1]],
	CatenateLayer[]
}, {
	NetPort["Input"] -> 7,
	NetPort["Input"] -> 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7
}];
getBlock2[i_] := NetChain@{
	getBN2[ToString@i], Ramp, getCN2[ToString@i, 0, 1],
	PoolingLayer[{2, 2}, "Stride" -> 2, "Function" -> Mean]
};
$getBlock2 = NetChain@{
	getBN2["4"], Ramp,
	PoolingLayer[{7, 7}, "Stride" -> 7, "Function" -> Mean]
};


(* ::Subchapter:: *)
(*Main*)


extractor = NetChain[{
	getCN2["0", 3, 2], 
	getBN2["0"], 
	ElementwiseLayer["ReLU"],
	PoolingLayer[{3, 3}, "Stride" -> 2, "PaddingSize" -> 1],
	NetChain@Table[getBlock[1, i], {i, 0, 10, 2}],
	getBlock2[1],
	NetChain@Table[getBlock[2, i], {i, 0, 22, 2}],
	getBlock2[2],
	NetChain@Table[getBlock[3, i], {i, 0, 46, 2}],
	getBlock2[3],
	NetChain@Table[getBlock[4, i], {i, 0, 30, 2}],
	$getBlock2
}]
classifier = LinearLayer[1000,
		"Weights" -> params["arg:densenet0_dense0_weight"],
		"Biases" -> params["arg:densenet0_dense0_bias"]
	];
mainNet = NetChain[{
	"Extractor" -> extractor,
	"Classifier" -> classifier,
	"Predictor" -> SoftmaxLayer[]
},
	"Input" -> encoder,
	"Output" -> decoder
]


(* ::Subchapter:: *)
(*Export Model*)


Export["DenseNet121 trained on ImageNet.WXF", mainNet]


(* ::Subchapter:: *)
(*Test*)


<< MachineLearning`;<< NeuralNetworks`;<< MXNetLink`;<< DeepMath`;
SetDirectory@NotebookDirectory[];DateString[]
netName = "DenseNet121 trained on ImageNet";
testName = "DenseNet121 tested on ImageNet ValidationSet";
test = TestReport[testName <> ".mt"]


(* ::Subitem:: *)
(*Mon 19 Nov 2018 15:11:46*)


(* ::Subchapter:: *)
(*Report*)


upload = ImportString["\
![High Precision Classification Curve.png](https://i.loli.net/2018/11/19/5bf26bf8f1979.png)
![Classification Curve.png](https://i.loli.net/2018/11/19/5bf26bf8f0fa1.png)
![Accuracy Rejection Curve.png](https://i.loli.net/2018/11/19/5bf26bff8e3dc.png)
![ConfusionMatrix.png](https://i.loli.net/2018/11/19/5bf26c00cc598.png)
", "Data"];
report = ClassificationBenchmark[analyze,
	DeepMath`Tools`TestReportAnalyze[test],
	"Image" -> AssociationThread[Rule @@ Transpose[StringSplit[#, {"![", "](", ")"}]& /@ upload]]
];
ClassificationBenchmark[testName, report]
