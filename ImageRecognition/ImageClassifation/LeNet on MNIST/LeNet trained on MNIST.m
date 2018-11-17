(* ::Package:: *)

SetDirectory@NotebookDirectory[];
<< NeuralNetworks`
<< MXNetLink`
<< DeepMath`
DateString[]


(* ::Subitem:: *)
(*Sat 17 Nov 2018 14:55:36*)


(* ::Subchapter:: *)
(*Main*)


mainNet = NetModel["LeNet Trained on MNIST Data"]


(* ::Subchapter:: *)
(*Export Model*)


Export["LeNet trained on MNIST.WXF", mainNet]


(* ::Subchapter:: *)
(*Test*)


test = TestReport["LeNet trained on MNIST.mt"]
ClassifyAnalyzeExport[analyze, test]
