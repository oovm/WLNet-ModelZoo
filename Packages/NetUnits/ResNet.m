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


(* ::Subsection:: *)
(*Additional*)
SetAttributes[
	{ },
	{Protected, ReadProtected}
]
End[]