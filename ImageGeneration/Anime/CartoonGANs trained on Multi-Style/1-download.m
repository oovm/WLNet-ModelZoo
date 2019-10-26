(* ::Package:: *)

SetDirectory@NotebookDirectory[];
MonitoredDownload = ResourceFunction["MonitoredDownload"];
If[
	!FileExistsQ@"Hayao.h5",
	MonitoredDownload[
		"https://github.com/penny4860/Keras-CartoonGan/raw/master/params/Hayao.h5",
		"Hayao.h5"
	]
];
If[
	!FileExistsQ@"Hosoda.h5",
	MonitoredDownload[
		"https://github.com/penny4860/Keras-CartoonGan/raw/master/params/Hosoda.h5",
		"Hosoda.h5"
	]
];
If[
	!FileExistsQ@"Paprika.h5",
	MonitoredDownload[
		"https://github.com/penny4860/Keras-CartoonGan/raw/master/params/Paprika.h5",
		"Paprika.h5"
	]
];
If[
	!FileExistsQ@"Shinkai.h5",
	MonitoredDownload[
		"https://github.com/penny4860/Keras-CartoonGan/raw/master/params/Shinkai.h5",
		"Shinkai.h5"
	]
]
