(* ::Package:: *)

SetDirectory@NotebookDirectory[];
Needs["MXNetLink`"]
Needs["NeuralNetworks`"]
DateString[]


(* ::Subitem:: *)
(*Thu 25 Oct 2018 20:51:12*)


(* ::Subchapter:: *)
(*Main*)


mainNet = NetModel["LeNet Trained on MNIST Data"]


(* ::Subchapter:: *)
(*Export Model*)


Export["LeNet trained on MNIST.WXF", mainNet]


(* ::Subchapter:: *)
(*Test*)


TestReport["LeNet trained on MNIST.mt"]


mainNet=Import@"LeNet trained on MNIST.WXF";
obj = ResourceObject["MNIST"];
trainingData = ResourceData[obj, "TrainingData"];
testData = ResourceData[obj, "TestData"];


cm=ClassifierMeasurements[mainNet,testData]


cm@{
"ClassMeanCrossEntropy",
"EvaluationTime",
"ClassRejectionRate",
	"Properties"



}


ClassifierMeasurements//PD


SetDirectory@NotebookDirectory[];
