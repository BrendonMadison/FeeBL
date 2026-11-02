(* ::Package:: *)
(* Generate all tree-level QED amplitudes for e+(p1) e-(p2) -> e+(p3) e-(p4) l+(p5) l-(p6),
   l = e (Landau-Lifshitz, 36 diagrams) and l = mu (12 diagrams), with FeynArts, and convert
   them to FeynCalc notation.  Both Feynman gauge and a general covariant gauge (xi kept) are
   saved; the latter is used for the gauge-invariance check of the numerical evaluator.     *)

$FeynCalcStartupMessages = False;
$LoadAddOns = {"FeynArts"};
<< FeynCalc`
$FAVerbose = 0;

outDir = DirectoryName[$InputFileName];
procs = <|
  "eeee" -> ({-F[1, {1}], F[1, {1}]} -> {-F[1, {1}], F[1, {1}], -F[1, {1}], F[1, {1}]}),
  "eemm" -> ({-F[1, {1}], F[1, {1}]} -> {-F[1, {1}], F[1, {1}], -F[1, {2}], F[1, {2}]})
|>;

tops = CreateTopologies[0, 2 -> 4];

convert[diags_, gaugeOpts___] := FCFAConvert[
  CreateFeynAmp[diags, gaugeOpts],
  IncomingMomenta -> {p1, p2}, OutgoingMomenta -> {p3, p4, p5, p6},
  ChangeDimension -> 4, List -> True, SMP -> True, DropSumOver -> True,
  UndoChiralSplittings -> True,
  FinalSubstitutions -> {SMP["e"] -> ee, SMP["m_e"] -> me, SMP["m_mu"] -> mmu}];

KeyValueMap[
  Function[{name, proc},
    Module[{diags, ampF, ampXi},
      diags = InsertFields[tops, proc, InsertionLevel -> {Particles}, Model -> "QED"];
      ampF = convert[diags];
      ampXi = convert[diags, GaugeRules -> {}];
      Put[ampF, FileNameJoin[{outDir, "amp_" <> name <> "_feynman.m"}]];
      Put[ampXi, FileNameJoin[{outDir, "amp_" <> name <> "_xi.m"}]];
      Print[name, ": ", Length[ampF], " diagrams; gauge symbols in xi version: ",
        Union[Cases[ampXi, _GaugeXi, Infinity]]];
      Print["  diagram 1 (Feynman gauge): ", InputForm[FCE[ampF[[1]]]]];
      Print["  diagram ", Length[ampF], " (Feynman gauge): ", InputForm[FCE[ampF[[-1]]]]];
    ]],
  procs];
