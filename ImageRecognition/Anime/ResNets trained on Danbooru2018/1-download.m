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
CheckDownload[
	"https://github.com/RF5/danbooru-pretrained/releases/download/v0.1/resnet18-3f77756f.pth",
	"resnet18.pth"
];
CheckDownload[
	"https://github.com/RF5/danbooru-pretrained/releases/download/v0.1/resnet34-88a5e79d.pth",
	"resnet34.pth"
];
CheckDownload[
	"https://github.com/RF5/danbooru-pretrained/releases/download/v0.1/resnet50-13306192.pth",
	"resnet50.pth"
];
CheckDownload[
	"https://github.com/RF5/danbooru-pretrained/raw/master/img/egpic2.jpg",
	"Test.jpg"
];
