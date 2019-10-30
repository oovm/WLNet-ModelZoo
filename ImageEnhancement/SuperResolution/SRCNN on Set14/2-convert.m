(* ::Package:: *)

(* ::Subchapter:: *)
(*Import Weights*)


SetDirectory@NotebookDirectory[];


$NCHW = TransposeLayer[{1<->4, 2<->3, 3<->4}];
getCN[i_, s_, p_] := ConvolutionLayer[
	"Weights" -> $NCHW@params@TemplateApply["/convolution2d_`1`/convolution2d_`1`_W:0", {i}],
	"Biases" -> params@TemplateApply["/convolution2d_`1`/convolution2d_`1`_b:0", {i}],
	"PaddingSize" -> p, "Stride" -> s
];


(* ::Subchapter:: *)
(*Main*)


params = Import["m_model_adam_new30.h5", "Data"];
mainNet = NetChain[
	{
		getCN[1, 1, 4], Ramp,
		getCN[2, 1, 0], Ramp,
		getCN[3, 1, 2]
	},
	"Input" -> {1, Automatic, Automatic},
	"Output" -> "Image"
];
Export["SRCNN-S trained on Set14.MAT", mainNet, "WXF", PerformanceGoal -> "Speed"]


(* ::Subchapter:: *)
(*Export Model*)


params = Import["3051crop_weight_200.h5", "Data"];
mainNet = NetChain[
	{
		getCN[1, 1, 4], Ramp,
		getCN[2, 1, 1], Ramp,
		getCN[3, 1, 2]
	},
	"Input" -> {1, Automatic, Automatic},
	"Output" -> "Image"
];
Export["SRCNN-L trained on Set14.MAT", mainNet, "WXF", PerformanceGoal -> "Speed"]
