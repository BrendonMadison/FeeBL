(* ::Package:: *)
(* Compare the C++ helicity-amplitude engine (src/qedBackgrounds/qedbg_tree.h) with the
   FeynArts/FeynCalc oracle (theory/mathematica) at points dumped by tests/unit/test_tree_engine.
   Input lines: proc  |A|^2(C++)  then (p+, p-, px, py) for p1..p6 with 36 significant digits.
   Usage: wolframscript -file compare_oracle.wl oracle_points.txt                        *)

$FeynCalcStartupMessages = False;
$LoadAddOns = {"FeynArts"};
<< FeynCalc`
$FAVerbose = 0;
here = DirectoryName[$InputFileName];
theory = FileNameJoin[{here, "..", "..", "theory", "mathematica"}];
Get[FileNameJoin[{theory, "me_reference.wl"}]];

wp = 60;
meV = SetPrecision[0.00051099895, wp]; mmuV = SetPrecision[0.1056583755, wp];
consts = {ee -> 1, me -> meV, mmu -> mmuV};
(* same-sign beams generated on the fly: e-(p1) e-(p2) -> e-(p3) e-(p4) l+(p5) l-(p6) *)
sameSign[fl_] := FCFAConvert[CreateFeynAmp[InsertFields[CreateTopologies[0, 2 -> 4],
     {F[1, {1}], F[1, {1}]} -> {F[1, {1}], F[1, {1}], -F[1, {fl}], F[1, {fl}]}, InsertionLevel -> {Particles}, Model -> "QED"]],
   IncomingMomenta -> {p1, p2}, OutgoingMomenta -> {p3, p4, p5, p6}, ChangeDimension -> 4, List -> True, SMP -> True,
   DropSumOver -> True, UndoChiralSplittings -> True, FinalSubstitutions -> {SMP["e"] -> ee, SMP["m_e"] -> me, SMP["m_mu"] -> mmu}];
ampMM = <|"mmee" -> sameSign[1], "mmmm" -> sameSign[2]|>;
Print["diagrams: e-e- -> e-e- e+e-: ", Length[ampMM["mmee"]], ",  e-e- -> e-e- mu+mu-: ", Length[ampMM["mmmm"]]];
terms = <|"eeee" -> prepareAmp[Get[FileNameJoin[{theory, "amp_eeee_feynman.m"}]], consts],
          "eemm" -> prepareAmp[Get[FileNameJoin[{theory, "amp_eemm_feynman.m"}]], consts],
          "mmee" -> prepareAmp[ampMM["mmee"], consts], "mmmm" -> prepareAmp[ampMM["mmmm"], consts]|>;
order = {p1, p2, p3, p4, p5, p6};
num[s_String] := SetPrecision[ToExpression[StringReplace[s, {"e" -> "*^", "E" -> "*^"}]], wp];

file = If[Length[$ScriptCommandLine] > 1, $ScriptCommandLine[[2]], FileNameJoin[{here, "oracle_points.txt"}]];
worst = 0;
Do[
  Module[{w, proc, cpp, lc, vecs, ref, dev},
    w = StringSplit[line];
    proc = w[[1]]; cpp = num[w[[2]]];
    lc = Partition[num /@ w[[3 ;;]], 4];
    vecs = ({(#[[1]] + #[[2]])/2, #[[3]], #[[4]], (#[[1]] - #[[2]])/2} &) /@ lc;
    ref = spinSummedM2[terms[proc], Thread[order -> vecs], order];
    dev = N[Abs[cpp/ref - 1]];
    worst = Max[worst, dev];
    Print[proc, "  oracle = ", ScientificForm[N[ref, 16], 16], "   C++ = ", ScientificForm[N[cpp, 16], 16],
      "   rel. dev. = ", ScientificForm[dev, 3]]],
  {line, Select[ReadList[file, String], StringLength[#] > 0 &]}];
Print["worst relative deviation C++ vs FeynCalc oracle: ", ScientificForm[worst, 3]];
