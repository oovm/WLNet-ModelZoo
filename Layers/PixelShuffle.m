(* ::Package:: *)

Input: ChannelT[$$InputChannels, TensorT[$$InputSize]]

Output: ChannelT[$$OutputChannels, TensorT[$$OutputSize]]

(*ReshapeParams: {$$InputChannels, $$InputSize, $$OutputChannels, $$OutputSize}*)

Parameters:
	$Scaled: ValidatedParameterT[checkInput]
	$$InputChannels: SizeT
	$$InputSize: SizeListT[2]
	$$OutputChannels: ComputedType[SizeT, $$InputChannels/$Scaled]
	$$OutputSize: ComputedType[SizeListT[2], $$InputSize*$Scaled]

AllowDynamicDimensions: True

(*ComputedType[SizeListT[2], computeSize[$$InputSize]]*)

oS[in_,s_]:=MaybeDyn[in/s];

oc[in_,s_]:=MaybeDyn[in*s];

checkInput[s_] := If[
	And[IntegerQ@s,Positive@s], s,
	FailValidation[GluonCVLayer, "Scaled should be a Positive Integer."]
];

checkOutput[s_] := If[
	IntegerQ@First@s, s,
	FailValidation[GluonCVLayer, "Illegal magnification."]
];

Writer: Function[
	input = GetInput["Input", "Batchwise"];
	index = SowNode["reshape", input, "shape" -> {0, -4, -1, #Scaled^2, 0, 0}];
	index = SowNode["reshape", index, "shape" -> {0, 0, -4, #Scaled, #Scaled, 0, 0}];
	index = SowNode["transpose", index, "axes" -> {0, 1, 4, 2, 5, 3}];
	index = SowNode["reshape", index, "shape" -> {0, 0, -3, -3}];
	SetOutput["Output", index]
]
