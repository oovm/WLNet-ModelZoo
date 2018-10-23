(* ::Package:: *)
(* ::Title:: *)
(*NetLayers*)
(* ::Subchapter:: *)
(*Introduce*)
$LoadingLayers::usage = "help.";
GluonCV`PixelShuffleLayer::usage = "help.";
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
(*defFromFile*)
defFromFile[file_, sfx_String] := Block[
	{def},
	def = AssociateTo[NeuralNetworks`Private`ReadDefinitionFile[file, "GluonCV`"], "Suffix" -> sfx];
	NeuralNetworks`DefineLayer[FileBaseName[file], def]
];

$LoadingLayers := Block[
	{layers},
	Needs["NeuralNetworks`"];
	layers = FileNames["*", FileNameJoin[{$GluonCVDirectory, "Kernel", "Layers"}]];
	Quiet@Table[defFromFile[file, "Layer"], {file, layers}];
	True
];

(*TODO:Add Installer*)


(* ::Subsection:: *)
(*Additional*)
SetAttributes[
	{ },
	{Protected, ReadProtected}
];
End[]