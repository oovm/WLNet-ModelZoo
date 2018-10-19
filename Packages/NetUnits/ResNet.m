(* ::Package:: *)
(* ::Subsection::Closed:: *)
(*Defines*)
ResBlockV2::usage = "";
(* ::Subsection::Closed:: *)
(*Main*)
Begin["`ResNet`"];
(* ::Subsubsection:: *)
(*ResNetV2*)
BN[p__ : Nothing] := BatchNormalizationLayer["Epsilon" -> 1*^-5, p]
CN[c_, k_, p_ : 1, s_ : 1] := ConvolutionLayer[
	c, {k, k}, "Biases" -> None,
	"PaddingSize" -> p, "Stride" -> s
];
ResSampleV2[c_Integer] := NetGraph[{
	BN[], Ramp,
	CN[c, 3, 1, 2],
	BN[], Ramp,
	CN[c, 3, 1, 1],
	CN[c, 3, 0, 2],
	ThreadingLayer[Plus]
}, {
	NetPort["Input"] -> 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 8,
	NetPort["Input"] -> 7 -> 8
}];
ResBasicV2[c_Integer] := NetGraph[{
	BN[], Ramp,
	CN[c, 3],
	BN[], Ramp,
	CN[c, 3],
	ThreadingLayer[Plus]
}, {
	NetPort["Input"] -> 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7,
	NetPort["Input"] -> 7
}];
ResBlockV2[c_Integer, n_Integer, head_ : True] := Block[
	{chain},
	chain = ConstantArray[ResBasicV2[c], n];
	If[head, PrependTo[chain, ResSampleV2[c]]];
	NetChain@chain
];


head := {
	BatchNormalizationLayer["Epsilon" -> 1*^-5],
	ConvolutionLayer[64, {7, 7}, "PaddingSize" -> 3, "Stride" -> 2],
	ElementwiseLayer["ReLU"],
	PoolingLayer[{3, 3}, "PaddingSize" -> 2, "Stride" -> 2]
};
tail := {
	BatchNormalizationLayer["Epsilon" -> 1*^-5],
	ElementwiseLayer["ReLU"],
	PoolingLayer[{7, 7}, "Function" -> Mean, "Stride" -> 7],
	FlattenLayer[]
};
predict := {1000, SoftmaxLayer[]};
NetChain@{
	head,
	ResBlockV2[64, 2, False],
	ResBlockV2[128, 4, True],
	ResBlockV2[256, 8, True],
	tail, predict
}

(* ::Subsection:: *)
(*Additional*)
SetAttributes[
	{ },
	{Protected, ReadProtected}
]
End[]