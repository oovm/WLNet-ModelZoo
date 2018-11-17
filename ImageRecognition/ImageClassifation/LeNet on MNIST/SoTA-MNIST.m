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





(* ::Section:: *)
(*Main*)


doTest["LeNet Trained on MNIST"]
