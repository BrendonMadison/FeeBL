(* ::Package:: *)
(* FeynArts/FeynCalc oracle for gamma(k1) e-(p1) -> e-(p2) l+(p3) l-(p4), l = e, mu, compared with the
   C++ engine (qedbg_tree.h, amplitude_2lines_photon) at points dumped by tests/unit/test_bh_bw_engine.
   Line format: proc |A|^2(C++) k(+,-,x,y) p1(+,-,x,y) ... p4(+,-,x,y), quad precision.
   Photon polarisations here: Coulomb gauge (eps^0 = 0, eps perpendicular to k), summed over two states. *)

$FeynCalcStartupMessages = False;
$LoadAddOns = {"FeynArts"};
<< FeynCalc`
$FAVerbose = 0;
here = DirectoryName[$InputFileName];
Get[FileNameJoin[{here, "..", "..", "theory", "mathematica", "me_reference.wl"}]];

wp = 60;
meV = SetPrecision[0.00051099895, wp]; mmuV = SetPrecision[0.1056583755, wp];
consts = {ee -> 1, me -> meV, mmu -> mmuV};
tops = CreateTopologies[0, 2 -> 3];
conv[proc_] := FCFAConvert[CreateFeynAmp[InsertFields[tops, proc, InsertionLevel -> {Particles}, Model -> "QED"]],
   IncomingMomenta -> {k1, p1}, OutgoingMomenta -> {p2, p3, p4}, ChangeDimension -> 4, List -> True,
   SMP -> True, DropSumOver -> True, UndoChiralSplittings -> True, TransversePolarizationVectors -> {k1},
   FinalSubstitutions -> {SMP["e"] -> ee, SMP["m_e"] -> me, SMP["m_mu"] -> mmu}];
amps = <|"egeee" -> conv[{V[1], F[1, {1}]} -> {F[1, {1}], -F[1, {1}], F[1, {1}]}],
         "egemm" -> conv[{V[1], F[1, {1}]} -> {F[1, {1}], -F[1, {2}], F[1, {2}]}]|>;
Print["diagrams: e gamma -> e e+e-: ", Length[amps["egeee"]], ",  e gamma -> e mu+mu-: ", Length[amps["egemm"]]];
terms = prepareAmp[#, consts] & /@ amps;
order = {p1, p2, p3, p4};
num[s_String] := SetPrecision[ToExpression[StringReplace[s, {"e" -> "*^", "E" -> "*^"}]], wp];
cart[lc_] := {(lc[[1]] + lc[[2]])/2, lc[[3]], lc[[4]], (lc[[1]] - lc[[2]])/2};
coulombPols[k_] := Module[{n = Normalize[k[[2 ;; 4]]], a, e1, e2},
   a = If[Abs[n[[3]]] < 0.9, {0, 0, 1}, {1, 0, 0}];
   e1 = Normalize[Cross[n, a]]; e2 = Cross[n, e1];
   {Join[{0}, e1], Join[{0}, e2]}];

file = If[Length[$ScriptCommandLine] > 1, $ScriptCommandLine[[2]], FileNameJoin[{here, "oracle_bh_points.txt"}]];
worst = 0;
Do[
  Module[{w, proc, cpp, vecs, kv, pols, rules, ref, dev},
    w = StringSplit[line]; proc = w[[1]]; cpp = num[w[[2]]];
    vecs = cart /@ Partition[num /@ w[[3 ;;]], 4];
    kv = vecs[[1]];
    pols = coulombPols[kv];
    rules = Join[Thread[order -> vecs[[2 ;;]]], {k1 -> kv}];
    ref = Sum[spinSummedM2[terms[proc], Join[rules, {Polarization[k1, ___] :> ep}], order], {ep, pols}];
    dev = N[Abs[cpp/ref - 1]]; worst = Max[worst, dev];
    Print[proc, "  oracle = ", ScientificForm[N[ref, 16], 16], "   C++ = ", ScientificForm[N[cpp, 16], 16], "   rel. dev. = ", ScientificForm[dev, 3]]],
  {line, Select[ReadList[file, String], StringLength[#] > 0 &]}];
Print["worst relative deviation C++ vs FeynCalc oracle (e gamma -> e l+l-): ", ScientificForm[worst, 3]];
