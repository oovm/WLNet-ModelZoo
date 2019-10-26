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
	"https://github.com/penny4860/Keras-CartoonGan/raw/master/params/Hayao.h5",
	"Hayao.h5"
];
CheckDownload[
	"https://github.com/penny4860/Keras-CartoonGan/raw/master/params/Hosoda.h5",
	"Hosoda.h5"
];
CheckDownload[
	"https://github.com/penny4860/Keras-CartoonGan/raw/master/params/Paprika.h5",
	"Paprika.h5"
];
CheckDownload[
	"https://github.com/penny4860/Keras-CartoonGan/raw/master/params/Shinkai.h5",
	"Shinkai.h5"
];
CheckDownload[
	"https://github.com/penny4860/Keras-CartoonGan/raw/master/sample_in/in1.png",
	"Test.png"
];
