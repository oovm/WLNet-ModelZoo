BeginTestSection["ClassificationBenchmark"];


(*Dependency Check*)
VerificationTest[
	<< MachineLearning`;
	<< NeuralNetworks`;
	<< MXNetLink`;
	<< DeepMath`;,
	Null, TestID -> "Dependency Check"
];


(*Pre-define*)
VerificationTest[
	model := model = Import[netName <> ".WXF"];
	data := data = Import@"D:\\WLNet-Data-Set\\ImageNet\\ILSVRC2012_ValidationData.WXF";
	cm := cm = ClassificationBenchmark[model, data];
	dump := dump = DumpSave[".cache.mx", cm];
	analyze := analyze = ClassificationBenchmark[cm, netName];,
	Null, TestID -> "Pre-define"
];


(*Warm-Up*)
VerificationTest[
	Print@With[{size = 1000},
		x = RandomReal[1, {size, size}];
		layer = NetInitialize@LinearLayer[size, "Input" -> size, "Biases" -> None];
		time = First@RepeatedTiming[layer[x, TargetDevice -> "GPU"]];
		Quantity[size^2 * (2 * size \[Minus] 1) / time, "FLOPS"]
	];,
	Null, TestID -> "GPU Warm-Up"
];


(*Evaluation*)
VerificationTest[Head[model], NetChain, TestID -> "Loading Model"];
VerificationTest[Head[data], List, TestID -> "Loading Data"];
VerificationTest[Head[cm], ClassifierMeasurementsObject, TestID -> "Benchmark Test"];
VerificationTest[Head[dump], List, TestID -> "Result Dump"];
VerificationTest[Head[analyze], Association, TestID -> "Analyzing"];

EndTestSection[];