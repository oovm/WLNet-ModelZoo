(* ::Package:: *)
(* ::Subsection::Closed:: *)
(*Defines*)
VggBasicBN::usage = "";
VggBasic::usage = "";
(* ::Subsection::Closed:: *)
(*Main*)
Begin["`VggNet`"];
VggBasic[c_Integer, u_Integer] := Block[
	{unit, pool},
	unit = {
		ConvolutionLayer[c, {3, 3}, "PaddingSize" -> 1, "Stride" -> 1],
		ElementwiseLayer["ReLU"]
	};
	pool = PoolingLayer[{2, 2}, "Stride" -> 2];
	NetChain@Flatten[{ConstantArray[unit, u], pool}]
];
VggBasicBN[c_Integer, u_Integer] := Block[
	{unit, pool},
	unit = {
		ConvolutionLayer[c, {3, 3}, "PaddingSize" -> 1, "Stride" -> 1],
		BatchNormalizationLayer["Epsilon" -> 1*^-5],
		ElementwiseLayer["ReLU"]
	};
	pool = PoolingLayer[{2, 2}, "Stride" -> 2];
	NetChain@Flatten[{ConstantArray[unit, u], pool}]
];


(* ::Subsection:: *)
(*Additional*)
SetAttributes[
	{ },
	{Protected, ReadProtected}
];
End[]