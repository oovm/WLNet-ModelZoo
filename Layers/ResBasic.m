(* ::Package:: *)

Input: InterleavingSwitchedT[$Interleaving, $$InputChannels, $$InputSize]

Output: InterleavingSwitchedT[$Interleaving, $OutputChannels, $$OutputSize]

Arrays:
	$Weights: TensorT[{$$InputChannels, $OutputChannels}, TensorT[$KernelSize]]
	$Biases: Nullable[VectorT[$OutputChannels]]

Parameters:
	$OutputChannels: 	SizeT
	$KernelSize:  		ArbSizeListT[2, PosIntegerT, None]
	$Stride: 			ArbSizeListT[2, PosIntegerT, 1]
	$PaddingSize: 		ArbSizeListT[2, NaturalT, 0]
	$Interleaving:		Defaulting[BooleanT, False]
	$$InputChannels: 	SizeT
	$$GroupNumber: 		Defaulting[SizeT, 1]
	$$InputSize: 		SizeListT[2]
	$$OutputSize: 		ComputedType[SizeListT[2], DeconvolutionShape[$$InputSize, $PaddingSize, $KernelSize, $Stride]]

ReshapeParams: {$$InputChannels, $$InputSize, $$OutputSize}

MinArgCount: 0
PosArgCount: 2

FinalCheck: Function[
	inputSize = $$InputSize;
	CheckConvolutionOrPoolingDynamic[inputSize, DeconvolutionLayer, $Input, $Interleaving, 0, $Stride];
	If[Min[$$OutputSize] < 1, FailValidation["choice of parameters results in a zero-size output tensor."]]
]

Writer: Function[
	If[#Interleaving === False, MXWriteDefaultAndReturn[]];
	input = GetInput["Input"];
	output = SowTransposedConvolutionOrPooling[2, input, #Weights, #Biases];
	SetOutput["Output", output];
]
