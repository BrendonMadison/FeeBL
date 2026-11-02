(* ::Package:: *)
(* Validation of the numerical |M|^2 evaluator (me_reference.wl):
   (1) 2->2: helicity-sum evaluator vs FeynCalc symbolic traces (FermionSpinSum + DiracSimplify)
       for e+e- -> mu+mu- and Bhabha, and vs the textbook massive e+e- -> mu+mu- formula;
   (2) 2->4 (e+e- -> e+e-e+e-, e+e- -> e+e-mu+mu-): gauge-parameter independence,
       Lorentz invariance, identical-particle and charge-conjugation symmetry.            *)

$FeynCalcStartupMessages = False;
$LoadAddOns = {"FeynArts"};
<< FeynCalc`
$FAVerbose = 0;
dir = DirectoryName[$InputFileName];
Get[FileNameJoin[{dir, "me_reference.wl"}]];

wp = 50;
meV = SetPrecision[0.00051099895, wp]; mmuV = SetPrecision[0.1056583755, wp];
consts = {ee -> 1, me -> meV, mmu -> mmuV};
relDev[a_, b_] := Abs[a - b]/Abs[b];
report[label_, a_, b_] := Print[StringPadRight[label, 58], " rel. dev. = ", ScientificForm[N[relDev[a, b]], 3]];

(* ---------------- (1) 2 -> 2 ---------------- *)
tops22 = CreateTopologies[0, 2 -> 2];
conv22[proc_] := FCFAConvert[CreateFeynAmp[InsertFields[tops22, proc, InsertionLevel -> {Particles}, Model -> "QED"]],
   IncomingMomenta -> {p1, p2}, OutgoingMomenta -> {p3, p4}, ChangeDimension -> 4, List -> True,
   SMP -> True, DropSumOver -> True, UndoChiralSplittings -> True,
   FinalSubstitutions -> {SMP["e"] -> ee, SMP["m_e"] -> me, SMP["m_mu"] -> mmu}];
traceM2[amp_] := Module[{a = Total[amp]},
   (a ComplexConjugate[a]) // FeynAmpDenominatorExplicit // FermionSpinSum[#, ExtraFactor -> 1] & // DiracSimplify];

point22[sqrts_, m_, mf_, cth_, phi_] := Module[{b, k},
   b = beams[sqrts, m]; k = decay[{sqrts, 0, 0, 0}, mf, mf, cth, phi];
   {p1 -> b[[1]], p2 -> b[[2]], p3 -> k[[1]], p4 -> k[[2]]}];
numPair[expr_, rules_] := expr /. consts /. Pair[Momentum[a_, ___], Momentum[b_, ___]] :> mdot[a /. rules, b /. rules];

Do[
  Module[{amp, tr, terms, pt, mf, vNum, vTr},
    {name, proc, mf} = cs;
    amp = conv22[proc];
    tr = traceM2[amp];
    terms = prepareAmp[amp, consts];
    pt = point22[SetPrecision[10, wp], meV, mf, SetPrecision[0.37, wp], SetPrecision[1.1, wp]];
    vNum = spinSummedM2[terms, pt, {p1, p2, p3, p4}];
    vTr = numPair[tr, pt];
    report[name <> " (" <> ToString[Length[amp]] <> " diagr.): helicity sum vs FeynCalc traces", vNum, vTr];
    If[name == "ee->mumu",
      With[{d = mdot[#1 /. pt, #2 /. pt] &},
        report["ee->mumu: helicity sum vs textbook massive formula", vNum,
          32/d[p1 + p2, p1 + p2]^2 (d[p1, p3] d[p2, p4] + d[p1, p4] d[p2, p3] + mmuV^2 d[p1, p2] +
             meV^2 d[p3, p4] + 2 meV^2 mmuV^2)]]]],
  {cs, {{"ee->mumu", {-F[1, {1}], F[1, {1}]} -> {-F[1, {2}], F[1, {2}]}, mmuV},
        {"Bhabha", {-F[1, {1}], F[1, {1}]} -> {-F[1, {1}], F[1, {1}]}, meV}}}];

(* ---------------- (2) 2 -> 4 ---------------- *)
order = {p1, p2, p3, p4, p5, p6};
SeedRandom[20261029];
Do[
  Module[{ampF, ampX, tF, tX, masses, sqrts, k, b, pt, v1, vs, sw, bv, ptB},
    {name, masses} = cs;
    ampF = Get[FileNameJoin[{dir, "amp_" <> name <> "_feynman.m"}]];
    ampX = Get[FileNameJoin[{dir, "amp_" <> name <> "_xi.m"}]];
    tF = prepareAmp[ampF, consts];
    sqrts = SetPrecision[91.1876, wp];
    Do[
      k = randomPoint4[sqrts, masses, wp];
      b = beams[sqrts, meV];
      pt = Thread[order -> Join[b, k]];
      v1 = spinSummedM2[tF, pt, order];
      Print[name, " point ", ip, ": sum|M|^2 = ", ScientificForm[N[v1], 12],
        "   (energy-momentum violation ", ScientificForm[N[Max[Abs[b[[1]] + b[[2]] - Total[k]]]], 2], ")"];
      Do[report["  gauge xi = " <> ToString[N[xi]] <> " vs Feynman gauge",
          spinSummedM2[prepareAmp[ampX /. GaugeXi[_] -> xi, consts], pt, order], v1],
        {xi, {SetPrecision[0, wp], SetPrecision[5.3, wp]}}];
      bv = SetPrecision[{0.31, -0.42, 0.55}, wp];
      ptB = pt /. (p_ -> v_) :> (p -> boostVec[v, bv]);
      report["  Lorentz boost (beta = 0.77)", spinSummedM2[tF, ptB, order], v1];
      report["  charge conjugation (p1<->p2, p3<->p4, p5<->p6)",
        spinSummedM2[tF, Thread[order -> (order[[{2, 1, 4, 3, 6, 5}]] /. pt)], order], v1];
      If[name == "eeee",
        report["  identical e-: p4 <-> p6", spinSummedM2[tF, Thread[order -> (order[[{1, 2, 3, 6, 5, 4}]] /. pt)], order], v1];
        report["  identical e+: p3 <-> p5", spinSummedM2[tF, Thread[order -> (order[[{1, 2, 5, 4, 3, 6}]] /. pt)], order], v1]],
      {ip, 2}]],
  {cs, {{"eeee", {meV, meV, meV, meV}}, {"eemm", {meV, meV, mmuV, mmuV}}}}];
