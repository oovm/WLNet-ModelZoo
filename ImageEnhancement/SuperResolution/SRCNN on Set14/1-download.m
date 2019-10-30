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
	"https://github.com/MarkPrecursor/SRCNN-keras/raw/master/3051crop_weight_200.h5",
	"3051crop_weight_200.h5"
];
CheckDownload[
	"https://github.com/MarkPrecursor/SRCNN-keras/raw/master/m_model_adam_new30.h5",
	"m_model_adam_new30.h5"
];