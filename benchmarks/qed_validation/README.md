# QED amplitude checks

This directory preserves the symbolic amplitudes, independent evaluators and
fixed phase-space points used to check the pair-production matrix elements.
The C++ tests compile the headers in FeeBL's `src/` directory. They do not use
a detached copy of the matrix-element implementation.

The original calculation files are identified by SHA256 in `provenance.json`.
Importing those files is not a fresh validation result. The integrated build
must pass its native tests, and any new symbolic evaluation must retain its
tool versions, output and exit status.

## Native checks

Configure and build FeeBL normally, then run:

```sh
ctest --test-dir build --output-on-failure -R '^feebl-qed-'
```

The integration and photon-domain tests each have two variants: `reference`
uses `-fno-fast-math -ffp-contract=off`; `production` explicitly uses
`-ffast-math`. They check proposal-support boundaries, finite work limits,
sampling exhaustion and the photon-angle domain. The input-contract test
checks the native parser's allowed combinations and invalid mode values.

`feebl-qed-tree-oracle` checks the Dirac equation, spinor completeness, current
identities, graph counts, gauge-current equivalence, Lorentz transformations
and identical-particle symmetry. `feebl-qed-bh-bw-oracle` additionally checks
the massive Breit–Wheeler analytic expression and external-photon Ward
identities. These two targets use GCC's `__float128`/libquadmath reference
arithmetic and are enabled on GNU x86-64 Unix builds. Their bounded generated
samples are algebraic tests, not collision ensembles or statistical cross-
section convergence studies. Other toolchains omit these two reference targets;
they must not report them as passed.

## Mathematica and FeynCalc

The symbolic scripts use Mathematica, FeynCalc and the FeynArts add-on. Install
those tools separately and make `wolframscript` available on the command line.
No proprietary runtime or third-party package is distributed here.

Run the following from the repository root:

```sh
wolframscript -file benchmarks/qed_validation/theory/mathematica/check_reference.wl
wolframscript -file benchmarks/qed_validation/examples/numerical/compare_oracle.wl
wolframscript -file benchmarks/qed_validation/examples/numerical/compare_oracle_bh.wl
```

The comparison scripts use their packaged point files by default. An explicit
point-file path may follow the script argument. `oracle_extreme_points.txt`
can be supplied to `compare_oracle.wl` for the additional small-angle cases.
The scripts report their worst relative discrepancy. Their original versions
print diagnostics without enforcing a pass threshold; inspect the numerical
result rather than interpreting a zero process exit as scientific success.
They are not run automatically by CTest.

`gen_amplitudes.wl` regenerates the four packaged amplitude files beside itself.
Run it in a separate copy of this directory when checking regeneration, so the
retained reference files remain available for comparison.

## Conventions and fixed data

Each point-file row starts with a process identifier and the C++ helicity-summed
squared amplitude, followed by light-cone components `(p+, p-, px, py)` in GeV.
Here `p+ = E + pz`, `p- = E - pz`, the metric is `(+,-,-,-)`, and the coupling
is set to one. Initial helicities are summed, not averaged; flux, phase-space,
coupling and initial-spin averaging factors belong to the cross-section
assembly rather than these amplitude-only values.

The retained algebraic fixtures use electron mass 0.00051099895 GeV and muon
mass 0.1056583755 GeV. These are explicit parameters of each fermion record and
the symbolic evaluator. They must remain the same on both sides of a comparison.
They need not equal the legacy transport constants used by a collision card;
production generation and transport must use one consistent mass definition.

The four-lepton files cover opposite-sign and same-sign electron beams, with
electron or muon final pairs. The Bethe–Heitler file lists the incoming real
photon first, then the four fermion momenta. The arbitrary test points carry no
luminosity weight and are not generated events for a beam-background plot.

These checks test the amplitudes and their implementation. They do not by
themselves validate the equivalent-photon approximation, finite-beam-size
model, photon timing, transport integration or a complete detector acceptance.
