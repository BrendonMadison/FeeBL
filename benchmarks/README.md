# Collision benchmarks and examples

These examples compare beam deflection, luminosity spectra, pair production,
and spin transport with published calculations, and explore horizontal-dispersion
monochromatization at 125 GeV. They provide numerical inputs and saved results
for the accompanying paper. They are not a convergence certificate for every
collider or observable.

The separate [FCC-ee CPU/GPU comparison](README.md)
contains sixteen measured input cards, native spectra, paired covariance and
timing records for the 240 GeV workload in the paper. Its saved results can be
reduced without a GPU. Running the full collision cards is a cluster-scale
performance study rather than an installation test.

## Data availability

Input cards, reference CSVs, the viewer example and current reduced results are
included. Full native records and historical plot inputs require the optional
[data archive](../docs/FeeBL_Manual_Paper.pdf). Restore it before using the full
replot and native-reduction commands below. New collisions and the standalone
QED/LL figure exporters do not need that archive.

## Run a small example

Build the CPU executable using the repository installation instructions. Python
3.10 or later is sufficient for the runner. From the repository root:

```sh
python benchmarks/run.py verify
python benchmarks/run.py list
python benchmarks/run.py run smoke --feebl build/src/feebl --output runs/smoke
```

The smoke example has 1,000 macroparticles per beam, a 32 by 32 transverse mesh,
eight longitudinal slices, and one substep. Radiation and spin evolution are
disabled. It checks setup and output generation, not physical accuracy.
`out.dat` contains the resolved beam parameters and integrated observables;
`stdout.txt`, `stderr.txt`, and `execution.json` preserve execution diagnostics.
`log.txt` concatenates the two native streams and contains the final backend
receipt required for GUI finalization. The card enables native GUI output.
Finalize and open this particular smoke run with:

```sh
python tools/finalize_gui_dataset.py runs/smoke \
  --binary build/src/feebl --card runs/smoke/acc.dat --source . \
  --full-crossing-angle-rad 0 --crossing-frequency-hz 14100
python tools/serve_viewer.py --run runs/smoke --open
```

A verified 15-frame smoke dataset is included for viewing without a new run:

```sh
python tools/serve_viewer.py --run benchmarks/example_output/gui_smoke --open
```

It contains both particle beams, field maps, frame diagnostics and spectra.
It is a setup example at reduced statistics, not a precision collider prediction.

The runner creates a new output directory and refuses to overwrite an existing
result. It uses one OpenMP/BLAS thread, a finite timeout, and an 8 GiB address-space
limit on POSIX systems. Override these with `--timeout SECONDS` and
`--memory-gib GIB`. The native binary must run on the same operating system as
Python; on Windows, run the Linux build and this script together inside WSL.
No cluster job is submitted by these commands.

## Published examples

`manifest.json` lists all cases, numerical settings, input checksums, and the
source revision that produced the saved results. `cases/<name>/acc.dat` is a
complete native input card. The bundle contains 71 cases. Examples include:

| Case or family | Physical purpose | Typical cost |
|---|---|---|
| `lep1994_focusing` | 45.6 GeV Bhabha-like probes, 31.3/51.6 mrad acceptance boundaries | Local diagnostic |
| `lep1994_focusing_zero` | Zero-field transport control with the same probes | Local diagnostic |
| `lep1994_focusing_transverse_refined` | Transverse-mesh and primary-sample control | Local refinement |
| `ilc500_ussc_pairs_2006` | Incoherent pairs, 500 GeV, production energy at least 5 MeV | Local diagnostic |
| `ilc1000_tdr_b1b_gaussian_proxy` | 1 TeV luminosity spectrum with approximate incoming Gaussian beams | Local diagnostic |
| `ilc_depol_baseline` | 2006 ILC nominal optics, full input spin alignment | Local diagnostic |
| `ilc_depol_beta20_40k` | Nominal optics used in later depolarization comparisons | Local refinement |
| `rimbault2009_*` | Low-N, Large-Y and Low-P depolarization optics | Local refinement |
| `geometry_d0`, `geometry_opposite`, `geometry_same` | Field-free Gaussian overlap and dispersion-sign closure | Local diagnostic |
| `halhf250_lf`, `halhf380`, `halhf550` | Gaussian reconstructions of HALHF design moments | Local diagnostic |
| `ilc_confirm_*`, `halhf_confirm_*` | Three-seed luminosity-retaining horizontal-dispersion study | Longer local run |
| `high_ilc_*`, `high_halhf_*` | High-dispersion 125 GeV scan, without a luminosity floor | Longer local run |
| `fcc_cw_*` | 125 GeV FCC-ee, 30 mrad crossing and baseline crab waist, 0â€“100 mm dispersion | Long, memory-intensive |

For example:

```sh
python benchmarks/run.py run ilc_depol_baseline \
  --feebl build/src/feebl --output runs/ilc_depol_baseline
```

All executable statements in historical cards are preserved; their headers have
been rewritten as scientific descriptions. A result from a later FeeBL revision
need not reproduce the same bytes or random-number consumption. Source revisions
are therefore retained separately from the current executable hash.

The LEP `event.ini` files are manufactured back-to-back electronâ€“positron probes,
not events from a radiative Bhabha generator. Their identity files record the
Gaussian/hourglass vertex sampler, time binding, coordinate conventions, and
the 2,048 correlated events. Probe coordinates are in native nm, energies in
GeV, and direction components are dimensionless. Each event has two tracks.

## Replot the saved comparisons

Install the small plotting dependencies and choose a new result directory:

```sh
python -m pip install -r benchmarks/requirements-analysis.txt
python benchmarks/analysis/replot.py --output results/benchmark_plots
```

This command recomputes 2,356 pulls in 16 comparison panels, writes numerical
pull summaries, and redraws the native dispersion scans and separate analytic
forecasts. It does not execute FeeBL or replace a seed ensemble. The same figures
are supplied in `example_output/plots`. Two high-dispersion cases also include
their genuine `out.dat` and fine luminosity-spectrum output in
`example_output/native`.
`example_output/benchmark_results.csv` contains the numerical comparison and
qualification table. To test the pull implementation:

```sh
python -m unittest discover -s benchmarks/analysis -p 'test_*.py' -v
```

The comparison CSV files retain both central values, bin edges, each variance
component, covariance, and uncertainty-model variations. Probability masses
refer to the common retained support; excluded graphical bins are not assigned
zero probability. Values labeled `pp` are polarization percentage points.
Per-beam helicity, a correlated spin-product loss, and effective polarization
are distinct quantities. The historical inter-code comparison uses the signed
design-axis components `q1 = S_z,1` and `q2 = -S_z,2`, with loss
`100 * (1 - sqrt(<q1*q2>_L))`. The separately labeled local-helicity diagnostic
uses `h_i = S_i dot p_hat_i` and `100 * (1 - sqrt(<h1*h2>_L))`.
Neither is the effective polarization. For an observable whose polarized
cross section has the usual opposite-helicity structure, correlated collision
samples require
`P_eff = (<h1>_L - <h2>_L) / (1 - <h1*h2>_L)`.
The familiar `P_eff = (P_minus-P_plus)/(1-P_minus*P_plus)` follows when the
helicity product factorizes into its two means. A collision-weighted sample
need not satisfy that condition. Apply the same energy selection and weights
to all three moments; a vanishing denominator does not define a finite
effective polarization.

Every pull is `(FeeBL-reference)/sqrt(V_FeeBL+V_reference-2*Cov)`. Its displayed
error is one, with endcaps. The denominator includes both datasets' uncertainty.
For independent unweighted counts the absolute uncertainty is `sqrt(N)` and
the relative uncertainty is `1/sqrt(N)`; weighted and correlated records require
a different sampling model. The historical single-seed CSVs use a
primary-macroparticle effective-count approximation. It overestimates realized
seed scatter and is retained only as historical evidence.
`analysis/comparison_statistics.py` replaces this proxy with covariance measured
across independent complete collision runs; the covariance of their mean is the
sample covariance divided by the seed count. It retains bin correlations and
never rescales errors to make pulls small. LEP means use conditional distinct-probe
errors. Published-curve
uncertainties are estimated from three-point linear contrasts, with local
pooling, a five-point linear-fit alternative, and a graphical-reading floor.
These estimates can include curvature and digitization effects. Their variation
is recorded separately; it is not substituted for the unit pull error bar.
The reference covariance is model-estimated, and a small collision ensemble has
a rank-deficient covariance. Neither the largest pull nor the sum of squared
pulls is presented as a formal significance or goodness-of-fit test.

## Regenerate the manuscript analyses

The complete-run sampling unit, empirical covariance, reference-trend estimator,
empty-bin rule and finite-ensemble calibration are described in
[`analysis/README.md`](analysis/README.md). Prepare a new native seed campaign
on the intended compute platform, then execute its frozen jobs through that
platform's scheduler. These commands do not submit jobs:

```sh
python benchmarks/analysis/seed_campaign.py prepare --output runs/seed_campaign --seeds 12
python benchmarks/analysis/seed_campaign.py execute --campaign runs/seed_campaign \
  --index 0 --feebl build/src/feebl
# Execute each remaining listed index once, then reduce all completed jobs.
python benchmarks/analysis/comparison_statistics.py reduce \
  --campaign runs/seed_campaign --output results/seed_vectors
python benchmarks/analysis/comparison_statistics.py build \
  --ensemble results/seed_vectors/seed_vectors.npz --output results/comparisons
python benchmarks/analysis/regenerate_paper.py \
  --comparisons results/comparisons --output results/paper
```

`results/paper` contains the manuscript's numerical PDF figures, PNG previews,
tabular LaTeX fragments and their source map. The analytical forecast and
independent-Bessel sampler oracle are executable analysis modules. Historical
study records remain distinct from newly executed native replicas.
The figure/table map identifies the native case, output and estimator;
`docs/paper/tables/input_parameters.csv` lists settings for every packaged card.

## Horizontal monochromatization and limits

The completed study changes horizontal dispersion, with matched zero-dispersion
controls and declared focusing choices. Vertical dispersion, longitudinal
dispersion/chirp, chromatic waist matching, and combined schemes are future
studies. They are not simulated by these example scans.

The native high-dispersion ILC/HALHF plots show observed peak FWHM, not the RMS of
the beamstrahlung tail. The uncertainty on repeated points is the standard error
of three independent seeds. The beta-convolved Gaussian response-width fits
remain subject to their numerical qualification checks. FCC-ee native width
estimates have strong binning dependence and are excluded from the physical
width comparison; its luminosity points are retained. The separate Gaussian
overlap forecast holds incoming moments fixed and omits disruption, radiation,
hourglass evolution, and detailed accelerator transport. It is not a realized
collider design. ILC uses an ideal bunch tilt at 14 mrad, HALHF a head-on candidate,
and FCC-ee the finite-angle baseline without RF bunch tilt. The native FCC-ee
scan stops at 100 mm.

The paper's Maxwell appendix and `refinement_plan.json` identify the larger
particle samples, independent seeds, and spatial/time refinements still needed.
Increase physical domain and resolution separately; a wider mesh at fixed cell
count can reduce field accuracy. Preserve paired controls and report all seeds.

## Reference data and attribution

`reference/sources.json` provides source titles, primary URLs/DOIs, figure/table
locations, column definitions, extraction methods and limitations. It covers
49 numerical datasets. Reference rows can be digitized histogram steps, vector
markers, raster ink cells, or table transcriptions; their number is not a Monte
Carlo event count. Original papers and third-party graphics are not bundled.

The principal comparisons use Voutsinas et al., *Phys. Lett. B* **800** (2020)
135068; Rimbault et al., *Phys. Rev. ST Accel. Beams* **9** (2006) 034402;
Moortgat-Pick et al., EPAC2006 WEPLS032; Rimbault, *J. Phys. Conf. Ser.* **198**
(2009) 012006; Bailey et al., PAC2011 WEP134; the FCAL review
arXiv:1411.4924; and the September 2026 HALHF Pre-CDR. The monochromatization
reference data include Zhang et al., *Nucl. Instrum. Meth. A* **1073** (2025)
170268. Retain attribution and the distinction between extracted numerical
facts and original figures when reusing these files. The repository's
distribution terms do not confer additional rights to the cited publications.

## Extended 125 GeV monochromatization scans

Forty additional cards are registered under `cases/monochromatization_125`.
The [scan guide](../examples/monochromatization/README.md) lists configurations,
resource requirements and the standalone figure command. These cards require
the parallel design-axis spectrum implementation in this source version.
Saved older results retain their original source identities.
