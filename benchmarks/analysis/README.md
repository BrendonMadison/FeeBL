# Numerical analysis and manuscript regeneration

These programs connect native outputs, published reference data, comparison
uncertainties and manuscript figures. Python 3.10 or later, NumPy, SciPy and
Matplotlib are required. The lightweight benchmark runner itself uses only the
Python standard library. No analysis command submits a scheduler job.

## Sampling covariance and pulls

`seed_campaign.py prepare` creates a new directory containing eight collision
families with twelve fresh seeds each. The cases are the three force-free
geometric controls, ILC 1 TeV, HALHF 250/380/550 GeV, and ILC 500 GeV pairs.
Only seeds and explicit checkpoint switches change. The campaign records native
input hashes. Each `execute` invocation runs exactly one finite-time solver
process with one numerical thread and records its binary hash, exit status,
output hashes and elapsed time. A repeated invocation cannot overwrite a run.
Execution success is not physical validation.

An interrupted campaign must not be reduced by discarding a failed seed.
`comparison_statistics.py inventory` reports every requested seed and receipt.
The explicit `reduce --complete-families-only` option includes only families
whose entire declared ensemble completed with one native binary. Its inventory
retains all failed and unexecuted seeds. `build --allow-partial` then records
omitted panels in `comparison_coverage.json`. These are partial results, not a
completed calibration campaign. A changed executable requires a new complete
ensemble for an affected family; do not mix old and repaired binaries within
one sampling covariance.

```sh
python benchmarks/analysis/seed_campaign.py prepare --output runs/covariance --seeds 12
python benchmarks/analysis/seed_campaign.py execute --campaign runs/covariance \
  --index 0 --feebl build/src/feebl
# Execute the other 95 indices once on the intended compute platform.
python benchmarks/analysis/comparison_statistics.py reduce \
  --campaign runs/covariance --output results/covariance
python benchmarks/analysis/calibrate_seed_covariance.py \
  --ensemble results/covariance/seed_vectors.npz --output results/covariance/calibration.json
python benchmarks/analysis/comparison_statistics.py build \
  --ensemble results/covariance/seed_vectors.npz --output results/comparisons
python benchmarks/analysis/regenerate_paper.py \
  --comparisons results/comparisons --output results/paper
```

The sampling unit is a complete independently seeded collision. For bin vector
`y_s`, the central value is the seed mean and its covariance is the unbiased
sample covariance divided by the number of seeds. Luminosity encounters reuse
particles and fields; their count, squared weights, and encounter Kish count
are not independent-collision counts. The historical primary-macroparticle
variance proxy is retained only for an explicit sensitivity comparison.

The pull is `(simulation-reference)/sqrt(V_sim+V_ref-2*C_sim,ref)`.
Simulation and published reference sources are treated as independent here.
Their separate diagonal variances are in each CSV, and their full bin
covariances are in companion NPZ files. Pull bars have length one on either
side and capped endpoints. These bars propagate the numerator variance with
the estimated denominator held fixed; they do not turn a small-sample
studentized statistic into a unit Gaussian variable.

`calibrate_seed_covariance.py` fixes the first eight seeds as training data and
holds out the remaining four. It reports training-versus-test mean differences
against a Student-t distribution with seven degrees of freedom, whose expected
RMS is `sqrt(7/5)`, not one.
The Student-t reference assumes approximately Gaussian complete-run estimates;
sparse or strongly skewed bins need more seeds. The same program tests three
fixed projections of each force-free Gaussian spectrum: mean energy offset,
second moment and an energy
window integral. Bin-level RMS and coverage fractions are descriptive because
bins are correlated. Small ensembles and estimated reference errors do not
justify a global sum-of-squared-pulls significance. No error rescaling is fitted
to the observed discrepancies.

Empty bins are retained. A normalized spectrum bin empty in every collision
receives a declared half-observation regularizer at the summed encounter
exposure, projected onto the fixed-total subspace. For an unnormalized pair
count averaged over `R` independent collisions, the corresponding variance is
`0.5 / R**2`. Each regularized bin is marked hollow and counted in the analysis
receipt. Its pull is a display value under that regularization, not evidence
that a tail has been measured. Larger event samples remain necessary.

Published-curve variance is reconstructed from three-point contrasts that
annihilate a local linear trend. Fifteen neighboring probes estimate the local
noise variance; the five-point construction is exported as a sensitivity
alternative. A declared graphical-reading floor remains where fluctuations
vanish. Normalization and interpolation propagate covariance. Curvature,
digitization and correlations in the original calculation can enter this
estimate; it is not a source-certified uncertainty. The three unlike pair
subprocess yields cannot be treated as neighboring points on a smooth curve.
Their reference error uses the explicit one-equivalent-crossing paired-daughter
scenario `V=2Y`, with `Y` and `4Y` alternatives reported. The original equivalent
exposure is unknown. No formal reference significance is inferred from it.

LEP errors are conditional on one simulated field: distinct probes, not
repeated trajectory rows, supply the variance. `reduce_lep_bias.py` pairs the
inner and outer probes and preserves their covariance. It applies the same
leading boundary formula to the published mean angles and the simulated ones.
The published radiation/acceptance result is listed separately.

## Figures, tables and retained study inputs

`regenerate_paper.py` creates `figures/`, `tables/`, `numerical_data/` and a
hash-bearing `generation.json`. It rebuilds the manuscript's numerical figures
and tabular-only LaTeX fragments, rather than the older standalone overview
panels drawn by `replot.py`. Captions, mathematical definitions and installation
examples remain authored manuscript text. The figure map is
`tables/paper_traceability.json`; `tables/input_parameters.csv` lists every
packaged input family and its settings.

The generation command performs these calculations:

| Producer | Input | Calculation |
|---|---|---|
| `comparison_statistics.py` | native seed outputs and primary reference CSVs | normalized means, full covariance and pulls |
| `sampler_oracle.py` | 120 independently seeded native sufficient-statistic records | exact-Bessel quadrature and trial SEMs |
| `tools/analyze_imported_spin.py` | three matched native imported-spin JSON files, losslessly compressed | paired event changes, conditional SEM and time-step sensitivity |
| `reduce_study_sources.py` | 30 compressed native scalar spectra | luminosity, paired density covariance, ratios, resonance overlaps, empirical FWHM |
| `project_gaussian_overlap.py` | SI beam parameters | 1,203 fixed-input projection points and independent spacetime-integral check |
| `reduce_pair_support.py` | native secondary records | occupied log-angle/log-pT cells, separately by species and final-z sign |
| `reduce_lep_bias.py` | paired LEP probe records | common-estimator luminosity bias and conditional SEM |
| `paper_tables.py` | the generated records and packaged cards | manuscript table bodies and source map |

The 12 confirmation response widths remain qualified outputs of the retained
beta-plus-delta convolved Gaussian fit, recorded in
`analysis_inputs/confirmation_fit_records.json`. Their means, ratios, SEMs and
Student-t intervals are recomputed. The fit itself is intentionally an explicit
bounded operation, not an implicit plotting step:

```sh
python benchmarks/analysis/study_fits/frozen_spectrum_analysis.py NATIVE_DIRECTORY \
  --nominal-W 125 --fit-budget-seconds 180 --output response_fit.json
```

The directory must contain `lumi.ee.out.design_axis_dLdW.fne` from a native run.
The retained full-family fitter, zero-delta/Gaussian nested limits, quadrature
checks, width profiles and runtime refusal are packaged together. A rejected
or timed-out fit never becomes an admitted response width. Replaying a fit can
change near a tolerance boundary with numerical-library versions; its quality
record must be checked. Empirical FWHM uses the native peak, nearest enclosing
half-height crossings and all predeclared two/four-bin grouping phases. It is
not substituted for a rejected fitted Gaussian width.

Selected-optics shape panels show ratios with paired-seed uncertainty because
the optics deliberately change the distribution. Historical display bin edges
are frozen; all six spectra and their covariance are recomputed on those
edges. Repeated high-dispersion points show seed SEMs; mesh changes and grouping
sensitivities remain separate numerical limitations. Native FCC-ee component
widths are excluded from the physical width comparison. The fragmentation
diagnostic retains their raw bins and fixed grouping, while luminosity points
remain in the comparison.

The rigid Gaussian projection includes the actual crossing angle in invariant
energy. It omits radiation, disruption, time-evolving waist/hourglass effects,
RF errors and transported optics. Nonlinear invariant RMS is a validity check,
not a replacement FWHM. At FCC-ee 100 mm the rigid luminosity ratio is 0.753,
versus 0.483 in the native study. Beyond 100 mm its curves are conditional
projections, not simulated machine performance. All three baseline assumptions
are named in the legend.

HALHF pair plots compare occupied support only. Digitized plot pixels are not
event yields. The mixed sample uses `halhf250_lf`; the matched incoherent-only
control uses `halhf250_ipc_control`. Both retain stored-particle thinning and
source hashes. Refinement must use physical weights if yield comparisons are
added.

The imported-spin figure and table are regenerated directly from the three
compressed native records in `example_output/imported_spin`. Record identity,
initial momenta and spin states must match across field-on, field-off and
time-refined runs. The three `bhabha_spin_*` cases share the exact source
fixtures. The paired event SEM is conditional on one sampled beam realization;
the observed time-step sensitivity remains a numerical limitation. These
manufactured probes do not provide a polarized Bhabha scattering prediction.

## Targeted refinements and subprocess checks

`prepare_domain_campaign.py prepare --output NEW_DIRECTORY` creates twelve
HALHF380/550 runs: transverse domain and cell count doubled or quadrupled,
with three baseline-matched seeds and fixed cell spacing. Longitudinal cells,
time steps, particle count and field-grid levels stay fixed. The `reduce`
action exports luminosity, photon numbers, energy losses and outside-grid
visit fractions/counts. These are main-grid/source-stencil rejections, not a
lost-charge fraction. Compare these to the same seeds in the baseline campaign.
This tests domain dependence; it is not a cell-spacing convergence test.

`campaign_summary.py` converts the compact `calibration.json`,
`baseline_scalars.csv` and `domain_scalars.csv` records into tabular-only LaTeX,
CSV and a source-hash receipt:

```sh
python benchmarks/analysis/campaign_summary.py \
  --compact results/campaign/compact --output results/campaign/tables
```

The covariance table reports descriptive held-out coverage, covariance rank
and the number of excluded zero-training-variance bins. A separate table tests
the three fixed analytic observables. Domain ratios use exactly matched seeds
and reject comparisons made with different native binaries. Their uncertainty
is the standard error across the three paired ratios; the rejection columns
are the largest native per-step fractions over those seeds.
For a partial campaign without completed domain controls, add
`--calibration-only`; no domain table is fabricated.

`breit_wheeler_oracle.py` supplies threshold-to-high-energy unpolarized
cross-section values and identifies the former normalization literal. The
normalization correction increases the cross section and therefore cannot
explain the historical positive yield excess. Independent seed scatter,
real-photon energy/overlap spectra, energy cuts and domain/grid changes must
be checked separately. Virtual-photon cutoffs cannot explain a real-real
Breit-Wheeler difference. Secondary orbit corrections affect angular
distributions, not the number of production trials.

The historical ILC pair card uses a 5 MeV daughter-energy cut, `beam_size=1` and
`pair_q2=2`. The latter applies the GUINEA-PIG virtuality choice
`Q_max^2=s/4`; CAIN used `Q_max=m_e` in the published comparison
([Rimbault et al., Sec. II.A.1](https://doi.org/10.1103/PhysRevSTAB.9.034402)).
This setting difference affects the virtual-photon channels, not Breit-Wheeler.

The current publication reduction uses `qed_figures/reduce_figures.py`. It uses
twelve seeds per model, `beam_size=3`, and compares effective daughter-production
cross sections normalized to the measured electron–positron luminosity. The
complete count/luminosity covariance is retained. Historical raw-count products
remain archived and are not substituted into the current paper.

The daughter count uses individual `pairs.dat` rows by process code, with
`pair_ratio=1`. `new_pair` applies the 5 MeV lab-frame total-energy threshold
separately to each daughter at production; one daughter can pass while the
other fails. Tracking does not reapply the cut, and no detector acceptance is
imposed here. In contrast, the native `n_BW`, `n_BH`, `n_LL` and `n_pairs`
summary counters are accumulated before daughter admission and must not be
substituted for these row counts. The published table describes electrons
plus positrons with a produced-lepton energy cut; it does not state a further
requirement that both daughters pass. The original codes' unpublished event
selection is not independently certified by this comparison.

When Compton production is enabled, dumped secondary process codes 0 and 1
are shared with real- and equivalent-photon Compton recoils. A mixed
`pairs.dat` file therefore cannot identify Breit–Wheeler or Bethe–Heitler
particles from that column alone. The ILC yield comparison requires
`do_compt=0`. HALHF angular-support plots distinguish the mixed charged
background from the separate Compton-off incoherent-pair control. The native
integrated Compton and pair-production counters remain separate.

`sampler_probe.cc` is the native fixed-field radiation driver. Its standalone
GNU/Clang build uses the same seven source files as the spin unit target:

```sh
c++ -O2 -std=c++17 -fno-fast-math -ffp-contract=off \
  -ffunction-sections -fdata-sections -I src -I build/src \
  benchmarks/analysis/sampler_probe.cc src/particlesCPP.cc src/rndmCPP.cc \
  src/tridentCPP.cc src/physicalTools.cc src/mathematicalTools.cc \
  src/besselCPP.cc src/splineCPP.cc -Wl,--gc-sections -o build/sampler_probe
```

Use the production compiler policy as a separate comparison when evaluating a
new native build. The arguments are `chi exposure s2 seed trials`.
`analysis_inputs/sampler_plan.json` lists every retained invocation and its
record hash. The packaged plan has five chi values, two exposures, three
transverse spin states and four repetitions, each with two million trials and
a unique seed. The oracle consumes its nine-column records. Emission and energy
observables within a run share trials; they are not independent tests.

The separate historical 240-check family includes 200 sampling means and 40
deterministic bounds checks. Its original records and a retrospective 5%
simultaneous bounded-variable assessment are packaged separately; run
`python benchmarks/analysis/historical_sampler_family.py --output results/historical_sampler`.
Its output records the retrospective criterion, assumptions and scope. It does not use the 120-stream
illustration or claim that the original shared seeds were independent.

Spatial resolution, time stepping, physical domain, particle count and seed
count address different errors. The manuscript's Maxwell appendix retains
their separate plans. A successful numerical unit test or a small pull is
not a general precision certificate.

## Selected publication figures

The following command replays the two 40-bin spectrum comparisons, fixed-field
sampler figure, two single-panel monochromatization spectra, historical ILC
polarization comparisons and historical HALHF angular populations:

```sh
python benchmarks/analysis/render_selected_figures.py --output results/selected_figures
```

The packaged result is `example_output/selected_figures`. Each spectrum is
remapped using bin overlaps and the complete simulation, reference and cross
covariance matrices before its pulls are computed. Reference gaps are retained;
there is no extrapolation. The peak display windows are 990–1010 GeV for ILC
and 242–252 GeV for HALHF. A display crop preserves the original panel
normalization. Display tables retain empty-bin regularizer provenance.

The ILC spectra use twelve independent collisions. The HALHF spectra retain
historical proxy covariance and are labelled historical; their small pulls
are not a calibrated agreement test of the repaired native code. The ILC peak
axis stops at the last retained bin edge, 1001.6 GeV; the wider 990–1010 GeV
input clipping window does not imply reference extrapolation. The angular
panels compare occupied support, not yields, and retain the historical mixed
Compton/pair population distinction. The polarization figure separates local
helicity from the correlated fixed-axis spin product. Its ILC-only coverage
does not substitute for the planned HALHF depolarization comparison.

## Independent Compton equation check

`compton_oracle.py` checks head-on, free-electron, unpolarized Compton
kinematics in light-cone coordinates. It compares the candidate integrated
Klein–Nishina primitive with a separate 60-digit quadrature of the rest-frame
angular cross section. The retained JSON lists the tested energy ranges,
mass convention, residual normalization, primary Geant4 manual equations and
scope. This checks the equations and their numerical evaluation; it does not
certify the native random generator or a simulated collision ensemble.

```sh
python -m pip install -r requirements-oracles.txt
python benchmarks/analysis/compton_oracle.py --output results/compton_oracle.json
```

The optional `mpmath` dependency is used only by this oracle. Ordinary figure
and table regeneration does not need it. The output path must be new. The
packaged reference is `example_output/analysis_inputs/compton_oracle_reference.json`.

## Interrupted campaigns and version controls

The reducer normally requires every requested family to be complete. Its
explicit `--complete-families-only` option retains the failed/unexecuted-job
inventory and excludes each incomplete family as a whole. A partial comparison
also requires `build --allow-partial`; missing panels are recorded, never filled
with selected successful seeds or historical error models.

The canonical seed blocks in `seed_campaign.py` are independent of case order
or subset selection. A recovery campaign must retain the originally declared
seeds, including the seed that exposed a failure. Complete affected families
are rerun under the corrected binary. Reusing an unaffected older family
requires an explicit cross-version control; `merge_seed_ensembles.py` checks
this receipt, disjoint families, all twelve canonical seeds and the per-family
native binary identities before assembling the eight-family comparison.

```sh
python benchmarks/analysis/merge_seed_ensembles.py \
  --inputs results/old/seed_vectors.npz results/repaired/seed_vectors.npz \
  --source-archives OLD_SOURCE_SHA256 REPAIRED_SOURCE_SHA256 \
  --controls results/cross_version_controls.json --output results/merged
```

The control receipt uses schema `feebl.cross-version-controls/1` and status
`PASS`. For every reused family it records the first canonical seed, both
binary hashes, matching old/new scientific-spectrum hashes and a successful exact
scientific-summary comparison. The only permitted summary metadata omission
is an explicitly recorded `imported_spin_mode=0` line. Raw spectrum hashes are
retained; a changed `#source_revision` line may be removed solely for the
additional payload-hash comparison, with both revision values recorded. No
scientific header or bin is removed. This establishes the
observed control behavior, not universal equivalence between source versions.

## Current focusing and pair ensembles

The current-paper commands require the verified 60-collision QED reduction at
`benchmarks/example_output/qed_figure_campaign/reduction`, or an explicit
`--qed-reduction` path. They stop if it is missing or changed; they do not
substitute older focusing or pair results. The input manifest and all 60 cards
belong under `benchmarks/cases/qed_figure_campaign`.

The commands below redraw figures and tables from the packaged reduction.
Recomputing that reduction requires the separate full native outputs and
receipts; the compact source package contains the reduced data.

```sh
python benchmarks/analysis/qed_figures/render_figures.py \
  --source . --reduction benchmarks/example_output/qed_figure_campaign/reduction \
  --output results/qed_figures
python benchmarks/analysis/qed_figures/render_tables.py \
  --source . --reduction benchmarks/example_output/qed_figure_campaign/reduction \
  --output results/qed_tables
```

The scripts verify input and producer hashes before rendering. Exported
tables and aliased figures preserve the numerical reduction and export
receipts. The separate HALHF luminosity-spectrum study retains its historical
source and covariance limitations. The luminosity-spectrum display ranges
are independent of this focusing/pair update.

