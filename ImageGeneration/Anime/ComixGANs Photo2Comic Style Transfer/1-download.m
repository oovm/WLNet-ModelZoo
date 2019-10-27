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
	"https://github.com/nijuyr/comixGAN/raw/master/data2/checkpoints/best/G_weights.best.hdf5",
	"ComixGAN Comic Style Transfer Alpha.hdf5"
];
CheckDownload[
	"https://github.com/nijuyr/comixGAN/raw/master/data2/checkpoints/comparison/comix_gan.h5",
	"ComixGAN Comic Style Transfer Beta.h5"
];
CheckDownload[
	"https://github.com/maciej3031/comixify/raw/master/ComixGAN/pretrained_models/generator_model2.h5",
	"ComixGAN Comic Style Transfer Gamma.h5"
];
CheckDownload[
	"https://github.com/maciej3031/comixify/raw/master/ComixGAN/pretrained_models/generator_model.h5",
	"ComixGAN Comic Style Transfer Delta.h5"
];
