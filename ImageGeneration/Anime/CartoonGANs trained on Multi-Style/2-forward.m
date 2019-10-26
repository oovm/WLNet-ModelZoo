(* ::Package:: *)

SetDirectory@NotebookDirectory[];
mainNet = Import@"CartoonGan trained on Hayao Style.WLNet"
mainNet = Import@"CartoonGan trained on Hosoda Style.WLNet"

img = ImageResize[ExampleData[{"TestImage", "Mandrill"}], 256]
newNet = NetReplacePart[mainNet, "Input" -> NetEncoder[{"Image", ImageDimensions@img}]];


newNet[img, TargetDevice -> "GPU"]
