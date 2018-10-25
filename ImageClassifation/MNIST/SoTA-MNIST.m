(* ::Package:: *)

(* ::Section:: *)
(*Preload*)


SetDirectory@NotebookDirectory[];
<< GluonCV`


(* ::Section:: *)
(*Prepare*)


obj = ResourceObject["MNIST"];
trData = ResourceData[obj, "TrainingData"];
tsData = ResourceData[obj, "TestData"];
doTest[name_] := Block[
	{net = Import[name <> ".WXF"], export},
	export = <|
		"NetInfomation" -> ClassificationInformation[net],
		"TrainingSet" -> ClassificationBenchmark[net, trData, {1, 2, 3, 5}],
		"TestingSet" -> ClassificationBenchmark[net, tsData, {1, 2, 3, 5}]
	|>;
	Export[name <> ".json", GeneralUtilities`ToAssociations@export]
];


(* ::Section:: *)
(*Warm-up*)


With[{size = 1000},
	x = RandomReal[1, {size, size}];
	layer = NetInitialize@LinearLayer[size, "Input" -> size, "Biases" -> None];
	time = First@RepeatedTiming[layer[x, TargetDevice -> "GPU"]];
	Quantity[size^2 * (2 * size \[Minus] 1) / time, "FLOPS"]
]


(* ::Section:: *)
(*Main*)


doTest["LeNet Trained on MNIST"]
