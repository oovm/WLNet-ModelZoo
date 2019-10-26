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
	"https://raw.githubusercontent.com/RF5/danbooru-pretrained/master/config/class_names_100.json",
	"class_names_100.ckpt.json"
];
CheckDownload[
	"https://raw.githubusercontent.com/RF5/danbooru-pretrained/master/config/class_names_500.json",
	"class_names_500.ckpt.json"
];
CheckDownload[
	"https://raw.githubusercontent.com/RF5/danbooru-pretrained/master/config/class_names_6000.json",
	"class_names_6000.ckpt.json"
];
