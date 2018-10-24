(* ::Package:: *)

SetDirectory@NotebookDirectory[];
Needs["MXNetLink`"]
Needs["NeuralNetworks`"]
DateString[]


(* ::Subitem:: *)
(*Wed 24 Oct 2018 22:56:19*)


(* ::Subchapter:: *)
(*Import Weights*)


params = NDArrayImport@"NTIRE2018_x8-0000.params";


(* ::Subchapter:: *)
(*Encoder & Decoder*)


encoder = NetEncoder[{"Image", {640, 360}}]
decoder = NetDecoder["Image"]


(* ::Subchapter:: *)
(*Pre-defined Structure*)


ndarray[n_] := params["arg:learned_" <> ToString[n]];
(*NetMapOperator[ParametricRampLayer["Slope"\[Rule]ndarray[n]]]*)
prelu[i_, n_] := ParametricRampLayer["Slope" -> Flatten@ConstantArray[Normal@ndarray[i], n]]


(* ::Subchapter:: *)
(*Main*)


mainNet = NetGraph[nodes, path, "Input" -> encoder, "Output" -> decoder]


(* ::Subchapter:: *)
(*Export Model*)
