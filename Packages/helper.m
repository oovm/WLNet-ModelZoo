(* ::Package:: *)
(* ::Title:: *)
(*helper*)
(* ::Subchapter:: *)
(*Introduce*)
helper::usage = "help.";
(* ::Subchapter:: *)
(*Main*)
(* ::Subsection:: *)
(*Settings*)
Begin["`helper`"];
Version$helper = "V1.0";
Updated$helper = "2018-10-19";
(* ::Subsection::Closed:: *)
(*Codes*)
(* ::Subsubsection:: *)
(*功能块 1*)
helper = True;
paraErr := Failure["InvalidRange", <|"Message" -> "Meaningless parameters"|>];



(* ::Subsubsection:: *)
(*功能块 2*)
urlHelp[url_] := ButtonBox[
	StyleBox[">>", "SR"], BaseStyle -> "Link",
	ButtonSource -> ButtonData, ButtonData -> url,
	ButtonFunction -> (SystemOpen[#]&)
]


(* ::Subsection:: *)
(*Additional*)
SetAttributes[
	{ },
	{Protected, ReadProtected}
];
End[]