(* ::Package:: *)
(* ::Title:: *)
(*NetUnits*)
(* ::Subchapter:: *)
(*Introduce*)
VggBlock::usage = "help.";
(* ::Subchapter:: *)
(*Main*)
(* ::Subsection:: *)
(*Settings*)
Begin["`NetUnits`"];
(* ::Subsection::Closed:: *)
(*主体代码*)
Version$NetUnits = "V1.0";
Updated$NetUnits = "2018-10-19";
(* ::Subsubsection:: *)
(*VggBlock*)
VggBlock[c_Integer,u_Integer:1,m_String:""]:=Block[
	{},
	If[Or[c<1,u<1],Return@GluonCV`helper`paraErr];
	Switch[m,
		"BN",VggBasicBN[c,u],
		___,VggBasic[c,u]
	]
];


(* ::Subsection:: *)
(*Additional*)
SetAttributes[
	{ },
	{Protected, ReadProtected}
];
End[]