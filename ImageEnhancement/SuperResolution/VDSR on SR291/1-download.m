(* ::Package:: *)

SetDirectory@NotebookDirectory[];
CheckDownload[link_, path_] := If[
	FileExistsQ@path,
	Return[],
	ResourceFunction["MonitoredDownload"][
		link, path,
		"IncludePlot" -> True,
		OverwriteTarget -> False
	];
];


CheckDownload[
	"https://github.com/WolframRhodium/Super-Resolution-Zoo/raw/master/VDSR/caffe-vdsr%40huangzehao/VDSR-0000.params",
	"VDSR-0000.params"
];
CheckDownload[
	"https://github.com/WolframRhodium/Super-Resolution-Zoo/raw/master/VDSR/caffe-vdsr%40huangzehao/VDSR-symbol.json",
	"VDSR-symbol.json"
];