BeginTestSection["ClassificationBenchmark"]

(*Evaluation*)
VerificationTest[
	netName = "LeNet trained on MNIST";
	testName = "LeNet Tested on MNIST TestSet";
	model := model = Import[netName <> ".WXF"];
	data := data = ResourceData[ResourceObject["MNIST"], "TestData"];
	cm := cm = ClassifierMeasurements[model, data];
	dump := dump = DumpSave[".cache.mx", cm];
	analyze := analyze = ClassifyAnalyze[<|"Name" -> testName, "Result" -> cm, "Net" -> model|>];,
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
VerificationTest[Head[cm], ClassifierMeasurementsObject, TestID -> "Benchmark Testing"];
VerificationTest[Head[dump], List, TestID -> "Result Dumping"];
VerificationTest[Head[analyze], Association, TestID -> "Analyzing"];

EndTestSection[];
