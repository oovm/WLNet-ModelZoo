Input: ChannelT[$$InputChannels, TensorT[$$InputSize]]

Output: ChannelT[$$OutputChannels, TensorT[$$OutputSize]]

Parameters:
	$Scaled: PosIntegerT
	$$InputChannels: SizeT
	$$InputSize: SizeListT[2]
	$$OutputChannels: ComputedType[SizeT, $$InputChannels / $Scaled^2]
	$$OutputSize: ComputedType[SizeListT[2], $$InputSize * $Scaled]

(*TODO: Final Check, output shape must integers*)

Writer: Function[
	input = GetInput["Input", "Batchwise"];
	index = SowNode["reshape", input, "shape" -> {0, -4, -1, #Scaled^2, 0, 0}];
	index = SowNode["reshape", index, "shape" -> {0, 0, -4, #Scaled, #Scaled, 0, 0}];
	index = SowNode["transpose", index, "axes" -> {0, 1, 4, 2, 5, 3}];
	index = SowNode["reshape", index, "shape" -> {0, 0, -3, -3}];
	SetOutput["Output", index]
]

Suffix: "Layer"