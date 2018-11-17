BeginTestSection["ClassifationBenchmark"]

(*Evaluation*)
VerificationTest[
	name = "ResnetV2-20 trained on CIFAR10";
	model := model = Import[name <> ".WXF"];
	data := data = Import@"D:\\WLNet-Data-Set\\CIFAR10\\CIFAR10 TestData.MX";
	cm := cm = ClassifierMeasurements[model, data];
	dump := dump = DumpSave[".cache.mx", cm];,
	Null, TestID -> "Pre-define"
];


(*Evaluation*)
VerificationTest[Head[model], NetChain, TestID -> "Loading Model"];
VerificationTest[Head[data], List, TestID -> "Loading Data"];
VerificationTest[Head[cm], ClassifierMeasurementsObject, TestID -> "Benchmark Testing"];
VerificationTest[Head[dump], List, TestID -> "Result Dump"];


EndTestSection[];
