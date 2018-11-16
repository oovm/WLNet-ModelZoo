BeginTestSection[name]

name = "LeNet trained on MNIST";
model := model = Import[name <> ".WXF"];
data := data = ResourceData[ResourceObject["MNIST"], "TestData"];
cm := cm = ClassifierMeasurements[model, testData];
dump := dump = Export[cm, name <> ".TestDataset.WXF"]



VerificationTest[Head[model], NetChain, TestID -> "Loading Model"];
VerificationTest[Head[data], List, TestID -> "Loading Data"];
VerificationTest[Head[cm], ClassifierMeasurementsObject, TestID -> "Testing"];
VerificationTest[Head[dump], String, TestID -> "Result Dump"];


EndTestSection[];
