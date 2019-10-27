Input: ChannelT[$$Channels, TensorT[$$InputDimensions]]
Output: ChannelT[$$Channels, TensorT[$$InputDimensions]]

Parameters:
	$Epsilon: Defaulting[ScalarT, 10^-8]
	$$Channels: SizeT
	$$InputDimensions: SizeListT[SizeT]

(*ATTENTION: there's no 'RSQRT' in ONNX*)
Writer: Function[
	input = GetInput["Input", "Batchwise"];
	path = SowNode["mean", SowSquare@input, "axis" -> 1, "keepdims" -> True];
	path = SowRSqrt@SowNode["_PlusScalar", path, "scalar" -> #Epsilon];
	output = SowNode["broadcast_mul", {input, path}];
	SetOutput["Output", output]
]

Suffix: "alizationLayer"