(* ::Package:: *)

SetDirectory@NotebookDirectory[];


<< DeepMath`;
DeepMath`NetMerge;

params = Import["resnet50.pth.wxf"];


$NCHW = TransposeLayer[{1<->4, 2<->3, 3<->4}];
getCN[name_, s_, p_] := ConvolutionLayer[
	"Weights" -> params[name <> ".weight"],
	"Biases" -> None,
	"PaddingSize" -> p, "Stride" -> s
];
getBN[name_] := BatchNormalizationLayer[
	"Biases" -> params[name <> ".bias"],
	"Scaling" -> params[name <> ".weight"],
	"MovingMean" -> params[name <> ".running_mean"],
	"MovingVariance" -> params[name <> ".running_var"],
	"Epsilon" -> 0.00001,
	"Momentum" -> 0.9
]
getLinear[name_, out_] := LinearLayer[
	out,
	"Weights" -> params[name <> ".weight"],
	"Biases" -> params[name <> ".bias"]
];


getBlock[name_] := GeneralUtilities`Scope[
	path = NetChain@{
		getCN[name <> ".conv1", 1, 0],
		getBN[name <> ".bn1"],
		Ramp,
		getCN[name <> ".conv2", 1, 1],
		getBN[name <> ".bn2"],
		Ramp,
		getCN[name <> ".conv3", 1, 0],
		getBN[name <> ".bn3"]
	};
	NetFlatten@NetChain@{NetMerge[path, Plus], Ramp}
]
getBlock2[name_, s_ : 2] := GeneralUtilities`Scope[
	left = NetChain@{
		getCN[name <> ".conv1", 1, 0],
		getBN[name <> ".bn1"],
		Ramp,
		getCN[name <> ".conv2", s, 1],
		getBN[name <> ".bn2"],
		Ramp,
		getCN[name <> ".conv3", 1, 0],
		getBN[name <> ".bn3"]
	};
	right = NetChain@{
		getCN[name <> ".downsample.0", s, 0],
		getBN[name <> ".downsample.1"]
	};
	NetFlatten@NetChain@{NetMerge[{left, right}, Plus], Ramp}
]


encoder = NetEncoder[{
	"Image", 320,
	"MeanImage" -> {0.713739812374115, 0.6627991795539856, 0.6518916487693787},
	"VarianceImage" -> {0.2969885468482971, 0.3017076551914215, 0.2979130446910858}^2
}]
decoder = NetDecoder[{"Class", Import["class_names_6000.ckpt.json"]}]
mainNet = NetChain[
	{
		{
			getCN["0.0", 2, 3],
			getBN["0.1"],
			Ramp
		},
		PoolingLayer[{3, 3}, 2, "PaddingSize" -> 1, "Function" -> Max],
		getBlock2["0.4.0", 1],
		Table[getBlock["0.4." <> ToString[i]], {i, 1, 2}],
		getBlock2["0.5.0", 2],
		Table[getBlock["0.5." <> ToString[i]], {i, 1, 3}],
		getBlock2["0.6.0", 2],
		Table[getBlock["0.6." <> ToString[i]], {i, 1, 5}],
		getBlock2["0.7.0", 2],
		Table[getBlock["0.7." <> ToString[i]], {i, 1, 2}],
		NetMerge[
			{AggregationLayer[Max], AggregationLayer[Mean]},
			Join,
			Expand -> True
		],
		{
			getBN["1.2"],
			getLinear["1.4", 512],
			Ramp
		},
		{
			getBN["1.6"],
			getLinear["1.8", 6000]
		},
		LogisticSigmoid
	}
]



image = Import["Test.jpg"]
mainNet = NetReplacePart[mainNet, {"Input" -> encoder, "Output" -> decoder}];
result = mainNet[image, "Probabilities"];
Take[ReverseSort@Select[result, # > 0.3&], UpTo[10]] // Dataset


NetInformation[mainNet, "LayerTypeCounts"] // ReverseSort // Dataset


export = <|
	"Main" -> mainNet,
	"Encoder" -> encoder,
	"Decoder" -> decoder
|>;
Export["ResNet-53 trained on Danbooru2018.MX", export, "WXF", PerformanceGoal -> "Speed"]
