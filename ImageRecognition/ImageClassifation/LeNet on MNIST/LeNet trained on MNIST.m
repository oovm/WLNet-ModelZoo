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


SetDirectory@NotebookDirectory[];DateString[]
test = TestReport["LeNet trained on MNIST.mt"]


(* ::Subchapter:: *)
(*Test Report*)


analyze = ClassificationBenchmark[cm, "LeNet trained on MNIST"];
final = ClassificationBenchmark[analyze,
	DeepMath`Tools`TestReportAnalyze[test],
	"Image" -> <|
		"Classification Curve.png" -> "https://i.loli.net/2018/11/17/5bf004151b12a.png",
		"High Precision Classification Curve.png" -> "https://i.loli.net/2018/11/17/5bf0041535eab.png",
		"Accuracy Rejection Curve.png" -> "https://i.loli.net/2018/11/17/5bf00415503a1.png",
		"ConfusionMatrix.png" -> "https://i.loli.net/2018/11/17/5bf0041545c4e.png"
	|>
];
ClassificationBenchmark["LeNet tested on MNIST TestSet", final]
