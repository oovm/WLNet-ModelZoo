Input : TensorT[$Dimensions]

Output : TensorT[$Dimensions]

Parameters :
	$Scaled : ValidatedParameterT[checkInput]
	$$Channels : SizeT
	$$InputSize : SizeListT[2]
	$$OutputSize : ComputedType[SizeListT[2], computeSize[$$InputSize, $Scaled]]



checkInput[s_] := If[
	And[IntegerQ@s, Positive@s], s,
	FailValidation[GluonCV, "Scaled should be a Positive Integer."]
];

computeSize[in_List, s_Integer] := Prepend[s Rest@in, First@in / s];

checkOutput[s_] := If[
	IntegerQ@First@s, s,
	FailValidation[GluonCVLayer, "Illegal magnification."]
];

Writer : Function[
	input = GetInput["Input", "Batchwise"];
	index = SowNode["reshape", input, "shape" -> {0, -4, -1, #Scaled^2, 0, 0}];
	index = SowNode["reshape", index, "shape" -> {0, 0, -4, #Scaled, #Scaled, 0, 0}];
	index = SowNode["transpose", index, "axes" -> {0, 1, 4, 2, 5, 3}];
	index = SowNode["reshape", index, "shape" -> {0, 0, -3, -3}];
	SetOutput["Output", index]
]