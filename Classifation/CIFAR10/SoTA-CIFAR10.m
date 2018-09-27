(* ::Package:: *)

(* ::Section:: *)
(*Preload*)


SetDirectory@NotebookDirectory[];
<< GluonCV`


(* ::Section:: *)
(*Prepare*)


obj = ResourceObject["CIFAR-10"];
trData = ResourceData[obj, "TrainingData"];
tsData = ResourceData[obj, "TestData"];
doTest[name_] := Block[
	{net = Import[name <> ".WXF"], export},
	export = <|
		"NetInfomation" -> ClassificationInformation[net],
		"TrainingSet" -> ClassificationBenchmark[net, trData, {1, 2, 3, 5}],
		"TestingSet" -> ClassificationBenchmark[net, tsData, {1, 2, 3, 5}]
	|>;
	Export[name <> ".json", export, "ExpressionJSON", "Compact" -> 3]
];


(* ::Section:: *)
(*Warm-up*)


With[{size = 1000} ,
	x = RandomReal[1, {size, size}];
	layer = NetInitialize@LinearLayer[size, "Input" -> size, "Biases" -> None];
	time = First@RepeatedTiming[layer[x, TargetDevice -> "GPU"]];
	Quantity[size^2 * (2 * size \[Minus] 1) / time, "FLOPS"]
]


(* ::Section:: *)
(*Main*)


doTest["Resnet20-V2 trained on CIFAR10"]
doTest["Resnet56-V2 trained on CIFAR10"]
doTest["Resnet110-V2 trained on CIFAR10"]


doTest["Resnet16-Wide10 trained on CIFAR10"]
doTest["Resnet28-Wide10 trained on CIFAR10"]
doTest["Resnet40-Wide8 trained on CIFAR10"]
