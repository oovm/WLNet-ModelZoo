Inputs:
	$Input: TensorT[$$Dimensions]
	$Target: TensorT[$$Dimensions]

Outputs:
	$Loss: ScalarT

Parameters:
	$$Dimensions: SizeListT[]

AllowDynamicDimensions: True

IsLoss: True

(*TODO: Final Check, output shape must integers*)

Writer: Function[
	MeanLossImplementation["L2"];
]

Suffix: "LossLayer"