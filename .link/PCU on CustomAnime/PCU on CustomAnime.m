raw = Import["model.h5", "Data"];
params[name_String] := Block[
	{$NCHW, prefix, input},
	$NCHW = TransposeLayer[{1<->4, 2<->3, 3<->4}];
	prefix = StringJoin["/model_weights/", First@StringSplit[name, "/"], "/"];
	input = raw[prefix <> name];
	Switch[
		Length@Dimensions@input,
		1, RawArray["Real32", input],
		4, RawArray["Real32", $NCHW[input]],
		_, RawArray["Real32", input]
	]
]
getCN[i_, p_, s_, ops___] := PartialConvolutionLayer[
	"Weights" -> params["p_conv2d_" <> ToString[i] <> "/img_kernel:0"],
	"Biases" -> params["p_conv2d_" <> ToString[i] <> "/bias:0"],
	"PaddingSize" -> p, "Stride" -> s, ops
]
getBN[i_Integer] := BatchNormalizationLayer[
	"Momentum" -> 0.99,
	"Beta" -> raw["/model_weights/EncBN" <> ToString[i] <> "/EncBN" <> ToString[i] <> "_3/beta:0"],
	"Gamma" -> raw["/model_weights/EncBN" <> ToString[i] <> "/EncBN" <> ToString[i] <> "_3/gamma:0"],
	"MovingMean" -> raw["/model_weights/EncBN" <> ToString[i] <> "/EncBN" <> ToString[i] <> "_3/moving_mean:0"],
	"MovingVariance" -> raw["/model_weights/EncBN" <> ToString[i] <> "/EncBN" <> ToString[i] <> "_3/moving_variance:0"]
];
getBN2[i_Integer] := BatchNormalizationLayer[
	"Momentum" -> 0.99,
	"Beta" -> params["batch_normalization_" <> ToString[i] <> "/beta:0"],
	"Gamma" -> params["batch_normalization_" <> ToString[i] <> "/gamma:0"],
	"MovingMean" -> params["batch_normalization_" <> ToString[i] <> "/moving_mean:0"],
	"MovingVariance" -> params["batch_normalization_" <> ToString[i] <> "/moving_variance:0"]
];
UpSampleImage = NetGraph[{
	ResizeLayer[Scaled /@ {2, 2}, "Resampling" -> "Nearest"],
	CatenateLayer[1]
}, {
	NetPort["1x"] -> 1,
	{NetPort["2x"], 1} -> 2 -> NetPort["Output"]
}];
UpSampleMask = NetGraph[{
	ResizeLayer[Scaled /@ {2, 2}, "Resampling" -> "Nearest"],
	CatenateLayer[1],
	AggregationLayer[Max, 1],
	ReplicateLayer[1](*Stupid!*)
}, {
	NetPort["1x"] -> 1,
	{NetPort["2x"], 1} -> 2 -> 3 -> 4 -> NetPort["Output"]
}];
mainNet = NetGraph[{
	"Enc_1" -> getCN[49, 3, 2],
	"Act_1" -> ElementwiseLayer["ReLU"],
	"Enc_2" -> getCN[50, 2, 2],
	"Act_2" -> {getBN[1], Ramp},
	"Enc_3" -> getCN[51, 2, 2],
	"Act_3" -> {getBN[2], Ramp},
	"Enc_4" -> getCN[52, 1, 2],
	"Act_4" -> {getBN[3], Ramp},
	"Enc_5" -> getCN[53, 1, 2],
	"Act_5" -> {getBN[4], Ramp},
	"Enc_6" -> getCN[54, 1, 2],
	"Act_6" -> {getBN[5], Ramp},
	"Enc_7" -> getCN[55, 1, 2],
	"Act_7" -> {getBN[6], Ramp},
	"Enc_8" -> getCN[56, 1, 2],
	"Act_8" -> {getBN[7], Ramp},
	"Iup_1" -> UpSampleImage,
	"Mup_1" -> UpSampleMask,
	"Dec_1" -> getCN[57, 1, 1],
	"Act^1" -> {getBN2[22], ElementwiseLayer[Ramp[#] - 0.2 * Ramp[-#]&]},
	"Iup_2" -> UpSampleImage,
	"Mup_2" -> UpSampleMask,
	"Dec_2" -> getCN[58, 1, 1],
	"Act^2" -> {getBN2[23], ElementwiseLayer[Ramp[#] - 0.2 * Ramp[-#]&]},
	"Iup_3" -> UpSampleImage,
	"Mup_3" -> UpSampleMask,
	"Dec_3" -> getCN[59, 1, 1],
	"Act^3" -> {getBN2[24], ElementwiseLayer[Ramp[#] - 0.2 * Ramp[-#]&]},
	"Iup_4" -> UpSampleImage,
	"Mup_4" -> UpSampleMask,
	"Dec_4" -> getCN[60, 1, 1],
	"Act^4" -> {getBN2[25], ElementwiseLayer[Ramp[#] - 0.2 * Ramp[-#]&]},
	"Iup_5" -> UpSampleImage,
	"Mup_5" -> UpSampleMask,
	"Dec_5" -> getCN[61, 1, 1],
	"Act^5" -> {getBN2[26], ElementwiseLayer[Ramp[#] - 0.2 * Ramp[-#]&]},
	"Iup_6" -> UpSampleImage,
	"Mup_6" -> UpSampleMask,
	"Dec_6" -> getCN[62, 1, 1],
	"Act^6" -> {getBN2[27], ElementwiseLayer[Ramp[#] - 0.2 * Ramp[-#]&]},
	"Iup_7" -> UpSampleImage,
	"Mup_7" -> UpSampleMask,
	"Dec_7" -> getCN[63, 1, 1],
	"Act^7" -> {getBN2[28], ElementwiseLayer[Ramp[#] - 0.2 * Ramp[-#]&]},
	"Iup_8" -> UpSampleImage,
	"Mup_8" -> UpSampleMask,
	"Dec_8" -> getCN[64, 1, 1],
	"Act^8" -> {
		ElementwiseLayer[Ramp[#] - 0.2 * Ramp[-#]&],
		ConvolutionLayer[
			"Weights" -> params["conv2d_4/kernel:0"],
			"Biases" -> params["conv2d_4/bias:0"],
			"PaddingSize" -> 0, "Stride" -> 1
		],
		ElementwiseLayer[LogisticSigmoid]
	}
},
	Flatten@{
		{NetPort["Input"], NetPort["InputMask"]} -> "Enc_1",
		NetPort["Enc_1", "Output"] -> "Act_1",
		{"Act_1", NetPort["Enc_1", "OutputMask"]} -> "Enc_2",
		NetPort["Enc_2", "Output"] -> "Act_2",
		{"Act_2", NetPort["Enc_2", "OutputMask"]} -> "Enc_3",
		NetPort["Enc_3", "Output"] -> "Act_3",
		{"Act_3", NetPort["Enc_3", "OutputMask"]} -> "Enc_4",
		NetPort["Enc_4", "Output"] -> "Act_4",
		{"Act_4", NetPort["Enc_4", "OutputMask"]} -> "Enc_5",
		NetPort["Enc_5", "Output"] -> "Act_5",
		{"Act_5", NetPort["Enc_5", "OutputMask"]} -> "Enc_6",
		NetPort["Enc_6", "Output"] -> "Act_6",
		{"Act_6", NetPort["Enc_6", "OutputMask"]} -> "Enc_7",
		NetPort["Enc_7", "Output"] -> "Act_7",
		{"Act_7", NetPort["Enc_7", "OutputMask"]} -> "Enc_8",
		NetPort["Enc_8", "Output"] -> "Act_8",
		
		{"Act_7", "Act_8"} -> "Iup_1",
		{NetPort["Enc_7", "OutputMask"], NetPort["Enc_8", "OutputMask"]} -> "Mup_1",
		{"Iup_1", "Mup_1"} -> "Dec_1",
		NetPort["Dec_1", "Output"] -> "Act^1",
		
		Table[{
			{"Act_" <> ToString[7 - i], "Act^" <> ToString[i]} -> "Iup_" <> ToString[i + 1],
			{NetPort["Enc_" <> ToString[7 - i], "OutputMask"], NetPort["Dec_" <> ToString[i], "OutputMask"]} -> "Mup_" <> ToString[i + 1],
			{"Iup_" <> ToString[i + 1], "Mup_" <> ToString[i + 1]} -> "Dec_" <> ToString[i + 1],
			NetPort["Dec_" <> ToString[i + 1], "Output"] -> "Act^" <> ToString[i + 1]
		}, {i, 6}
		],
		{NetPort["Input"], "Act^7"} -> "Iup_8",
		{NetPort["InputMask"], NetPort["Dec_7", "OutputMask"]} -> "Mup_8", {"Iup_8", "Mup_8"} -> "Dec_8",
		NetPort["Dec_8", "OutputMask"] -> NetPort["OutputMask"],
		NetPort["Dec_8", "Output"] -> "Act^8" -> NetPort["Output"]
	},
	"Input" -> NetEncoder[{"Image", 256}],
	"InputMask" -> NetEncoder[{"Image", 256, ColorSpace -> "Grayscale"}],
(*"Output"->NetDecoder["Image"],*)
	"OutputMask" -> NetDecoder["Image"]
]