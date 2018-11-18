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


<< MachineLearning`;<< NeuralNetworks`;<< MXNetLink`;<< DeepMath`;
SetDirectory@NotebookDirectory[];DateString[]
test = TestReport["LeNet trained on MNIST.mt"]


(* ::Subitem:: *)
(*Sun 18 Nov 2018 19:48:27*)


(* ::Subchapter:: *)
(*Test Report*)


upload = ImportString["\
![Classification Curve.png](https://i.loli.net/2018/11/17/5bf004151b12a.png)
![High Precision Classification Curve.png](https://i.loli.net/2018/11/17/5bf0041535eab.png)
![Accuracy Rejection Curve.png](https://i.loli.net/2018/11/17/5bf00415503a1.png)
![ConfusionMatrix.png](https://i.loli.net/2018/11/17/5bf0041545c4e.png)
", "Data"];
report = ClassificationBenchmark[analyze,
	DeepMath`Tools`TestReportAnalyze[test],
	"Image" -> AssociationThread[Rule @@ Transpose[StringSplit[#, {"![", "](", ")"}]& /@ upload]]
];
ClassificationBenchmark["LeNet tested on MNIST TestSet", report]
