(* ::Package:: *)
(* High-precision numerical evaluation of tree-level FeynCalc amplitudes.
   Spinor chains (DOT[Spinor, ..., Spinor]) are evaluated with explicit 4x4 Dirac matrices
   (Dirac representation) and the spin sum is done as a helicity sum:
       Sum_spins |M|^2 = Sum_{s1..sn} |Sum_diagrams A(s1..sn)|^2 .
   Any orthonormal spin basis works for the sum; u and v satisfy
       Sum_s u ubar = pslash + m ,  Sum_s v vbar = pslash - m .
   Requires FeynCalc to be loaded.  Momenta are given as {E, px, py, pz} via rules p_i -> vec. *)

g0 = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, -1, 0}, {0, 0, 0, -1}};
sig = {PauliMatrix[1], PauliMatrix[2], PauliMatrix[3]};
gUp = Join[{g0}, Table[ArrayFlatten[{{0, sig[[i]]}, {-sig[[i]], 0}}], {i, 3}]];  (* gamma^mu *)
metricDiag = {1, -1, -1, -1};
id4 = IdentityMatrix[4];
mdot[a_, b_] := a[[1]] b[[1]] - a[[2 ;; 4]] . b[[2 ;; 4]];
slash[k_] := gUp[[1]] k[[1]] - gUp[[2]] k[[2]] - gUp[[3]] k[[3]] - gUp[[4]] k[[4]];

uSpinor[p_, m_, s_] := Join[Sqrt[p[[1]] + m] UnitVector[2, s],
   (p[[2 ;; 4]] . sig) . UnitVector[2, s]/Sqrt[p[[1]] + m]];
vSpinor[p_, m_, s_] := Join[(p[[2 ;; 4]] . sig) . UnitVector[2, s]/Sqrt[p[[1]] + m],
   Sqrt[p[[1]] + m] UnitVector[2, s]];
barOf[x_] := Conjugate[x] . g0;

(* FeynCalc convention: Spinor[p, m] = u(p), Spinor[-p, m] = v(p); leftmost spinor is barred *)
spinorInfo[Spinor[Times[-1, Momentum[p_, ___]], m_, ___]] := {p, m, True};
spinorInfo[Spinor[Momentum[Times[-1, p_], ___], m_, ___]] := {p, m, True};
spinorInfo[Spinor[Momentum[p_, ___], m_, ___]] := {p, m, False};

(* a DOT element is linear in DiracGamma objects: c0 + sum_j c_j gamma_j *)
elemMatrix[el_, iv_, momRules_] := Module[{x, c0},
   x = Expand[el /. DiracGamma[a_, ___] :> dgm[a]];
   c0 = x /. dgm[_] -> 0;
   ((x - c0) /. {dgm[LorentzIndex[i_, ___]] :> gUp[[iv[i] + 1]],
       dgm[Momentum[k_, ___]] :> slash[k /. momRules]}) + c0 id4];

chainArray[ch_, momRules_] := Module[{els, mids, iL, iR, rows, cols, ids},
   els = List @@ ch; mids = Most[Rest[els]];
   iL = spinorInfo[First[els]]; iR = spinorInfo[Last[els]];
   rows = Table[barOf[If[iL[[3]], vSpinor, uSpinor][iL[[1]] /. momRules, iL[[2]], s]], {s, 2}];
   cols = Table[If[iR[[3]], vSpinor, uSpinor][iR[[1]] /. momRules, iR[[2]], s], {s, 2}];
   ids = Union[Cases[mids, LorentzIndex[i_, ___] :> i, Infinity]];
   <|"parts" -> {iL[[1]], iR[[1]]}, "ids" -> ids,
     "arr" -> Association[Table[a -> With[{mat = Dot @@ (elemMatrix[#, AssociationThread[ids -> a], momRules] & /@ mids)},
           Table[rows[[s1]] . mat . cols[[s2]], {s1, 2}, {s2, 2}]], {a, Tuples[Range[0, 3], Length[ids]]}]]|>];

scalarNum[expr_, iv_, momRules_] := expr /. {
    Pair[Momentum[a_, ___], Momentum[b_, ___]] :> mdot[a /. momRules, b /. momRules],
    Pair[LorentzIndex[i_, ___], Momentum[k_, ___]] :> (k /. momRules)[[iv[i] + 1]],
    Pair[Momentum[k_, ___], LorentzIndex[i_, ___]] :> (k /. momRules)[[iv[i] + 1]],
    Pair[LorentzIndex[i_, ___], LorentzIndex[j_, ___]] :> If[iv[i] == iv[j], metricDiag[[iv[i] + 1]], 0]};

(* one product term -> array over the spins of all external particles (slots in 'order') *)
termArray[term_, momRules_, order_] := Module[{factors, chains, scal, cdata, ids, perm},
   factors = If[Head[term] === Times, List @@ term, {term}];
   chains = Select[factors, Head[#] === DOT &];
   scal = Times @@ Select[factors, Head[#] =!= DOT &];
   cdata = chainArray[#, momRules] & /@ chains;
   ids = Union[Flatten[Join[#["ids"] & /@ cdata, Cases[scal, LorentzIndex[i_, ___] :> i, Infinity]]]];
   perm = Flatten[(Flatten[Position[order, #]] & /@ #["parts"]) & /@ cdata];
   Sum[With[{iv = AssociationThread[ids -> a]},
       (Times @@ (metricDiag[[# + 1]] & /@ a)) scalarNum[scal, iv, momRules] *
        Transpose[TensorProduct @@ (#["arr"][Lookup[iv, #["ids"]]] & /@ cdata), perm]],
     {a, Tuples[Range[0, 3], Length[ids]]}]];

(* prepare a list of FeynCalc diagram expressions for numerical evaluation *)
prepareAmp[amp_List, constRules_] := Module[{x},
   x = MomentumCombine[Contract[FeynAmpDenominatorExplicit[#]]] & /@ amp;
   x = x /. constRules;
   Flatten[(If[Head[#] === Plus, List @@ #, {#}] &[Expand[#]]) & /@ x]];

(* spin-summed |M|^2 at one phase-space point: momRules = {p1 -> {E,px,py,pz}, ...} *)
ampArray[terms_List, momRules_, order_] := Total[termArray[#, momRules, order] & /@ terms];
spinSummedM2[terms_List, momRules_, order_] := Total[Abs[Flatten[ampArray[terms, momRules, order]]]^2];

(* ---------- kinematics helpers ---------- *)
boostVec[p_, b_] := Module[{b2 = b . b, g, bp},
   g = 1/Sqrt[1 - b2]; bp = b . p[[2 ;; 4]];
   Join[{g (p[[1]] + bp)}, p[[2 ;; 4]] + ((g - 1) bp/b2 + g p[[1]]) b]];
twoBodyMom[M_, m1_, m2_] := Sqrt[(M^2 - (m1 + m2)^2) (M^2 - (m1 - m2)^2)]/(2 M);
decay[P_, m1_, m2_, cth_, phi_] := Module[{M, q, sth, k1, k2, b},
   M = Sqrt[mdot[P, P]]; q = twoBodyMom[M, m1, m2]; sth = Sqrt[1 - cth^2];
   k1 = {Sqrt[q^2 + m1^2], q sth Cos[phi], q sth Sin[phi], q cth};
   k2 = {Sqrt[q^2 + m2^2], -k1[[2]], -k1[[3]], -k1[[4]]};
   b = P[[2 ;; 4]]/P[[1]];
   If[b . b == 0, {k1, k2}, {boostVec[k1, b], boostVec[k2, b]}]];
(* e+(p1) along -z, e-(p2) along +z *)
beams[sqrts_, m_] := With[{e = sqrts/2}, {{e, 0, 0, -Sqrt[e^2 - m^2]}, {e, 0, 0, Sqrt[e^2 - m^2]}}];
(* random 2 -> 4 point by sequential two-body decays, working precision wp *)
randomPoint4[sqrts_, {m3_, m4_, m5_, m6_}, wp_] := Module[{r, M1, M2, P, k3, r1, k4, r2, k5, k6},
   r[a_, b_] := RandomReal[{a, b}, WorkingPrecision -> wp];
   M1 = r[m4 + m5 + m6, sqrts - m3]; M2 = r[m5 + m6, M1 - m4];
   P = {sqrts, 0, 0, 0};
   {k3, r1} = decay[P, m3, M1, r[-1, 1], r[0, 2 Pi]];
   {k4, r2} = decay[r1, m4, M2, r[-1, 1], r[0, 2 Pi]];
   {k5, k6} = decay[r2, m5, m6, r[-1, 1], r[0, 2 Pi]];
   {k3, k4, k5, k6}];
