(* ::Package:: *)

(* ::Subchapter:: *)
(*Import Weights*)


SetDirectory@NotebookDirectory[];
NetModel[];
<< DeepMath`
DeepMath`NetMerge;
NeuralNetworks`Private`MXNetFormat`readLayerCustom["broadcast_add", _] := ThreadingLayer[Plus]


raw = Import["VDSR-symbol.json", "MXNet"]


(* ::Subchapter:: *)
(*Main*)


path = Values@Normal@NetTake[raw, {"conv1", "conv20"}];
new = NetChain@Join[Partition[Most@path, 2], {Last@path}];
mainNet = NetMerge[new, Plus]


(* ::Subchapter:: *)
(*Export Model*)


Export["VDSR trained on SR291.MAT", mainNet, "WXF", PerformanceGoal -> "Speed"]
