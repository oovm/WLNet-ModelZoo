GluonCV::usage = "";
$GluonCVDirectory::usage = "";
$GluonCVData::usage = "";
Begin["`Private`"];
$GluonCVDirectory = DirectoryName[FindFile["GluonCV`Kernel`"], 2];
$GluonCVData = FileNameJoin[{$UserBaseDirectory, "ApplicationData", "GluonCV"}];


GluonCV = <|
	"Helper"->TrueQ@GluonCV`helper
	(*"Layers"->TrueQ@$LoadingLayers*)
|>;
End[]