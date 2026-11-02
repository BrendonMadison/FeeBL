# Horizontal-dispersion examples

## 125 GeV scans

The 40 cards in [benchmarks/cases/monochromatization_125](../../benchmarks/cases/monochromatization_125)
include the extended ILC/LCF and HALHF scans, FCC-ee at 100, 125, 150 and
175 mm, and their numerical control configurations. The portable
[catalogue](../../benchmarks/cases/monochromatization_125/catalogue.json)
records seeds and exact card hashes. All cards use accelerator block
`mono125` and parameter block `diagnostic`.

| Family | Production dispersion [mm] | Independent seeds per point | Peak FWHM [MeV] |
|---|---|---:|---|
| ILC/LCF | 75.865, 151.731 | 3 | 1.0695, 0.5964 |
| HALHF | 93.629, 187.258 | 3 | 1.0276, 0.5405 |
| FCC-ee | 100, 125, 150, 175 | 3 | 81.143, 71.225, 63.295, 56.985 |

Widths are observed design-axis histogram FWHMs before ISR. The linear
collider luminosity convention is the ILC 5 Hz rate; changing only the rate
to 10 Hz doubles luminosity. FCC-ee retains its finite-angle crab waist,
without RF bunch rotation. Storage-ring equilibrium is not recomputed.

The production GPU cards use one GPU and 16 host workers per collision,
empty-cell skipping, and enlarged grids. They are large scientific workloads.
The FCC-ee production grid is 2048 x 4096 x 128 with 80,000 macroparticles
per beam; its V100 run required about 22 GB device memory and 39 GB host
memory. Allow additional host headroom (64 GB or more per collision).
Some ILC/HALHF controls use still larger horizontal grids. The CPU16 controls
are identified by their names; the optional `serial1` card is a serial
comparison input, not a recommended production configuration.

```sh
python benchmarks/run.py verify
python benchmarks/run.py run ilc_fwhm05_fine_s1 --feebl build/src/feebl \
  --output runs/ilc_narrow --timeout 28800 --memory-gib 0
python benchmarks/run.py run fcc_cw_d175_highres_s1 --feebl build/src/feebl \
  --output runs/fcc175 --timeout 28800 --memory-gib 0
```

These commands run a collision; they do not allocate scheduler resources.
`--memory-gib 0` leaves memory limits to the scheduler or operating system.
The runner otherwise imposes an 8 GiB virtual-address limit on POSIX, which
is unsuitable for these large CUDA jobs. Reserve physical host and device
memory through the scheduler; a virtual-address limit is not an RSS limit.
Use a CUDA build for GPU cards and allocate one device and 16 physical CPU
cores per process. Avoid inherited `OMP_PROC_BIND`, `OMP_PLACES` or
`GOMP_CPU_AFFINITY` settings that pin all pool threads to one core.
Setting `OMP_NUM_THREADS=1` limits incidental OpenMP work; the cards explicitly
request the 16-worker host pool. Output directories must be new.

The compact spectra and updated tables are in
[benchmarks/example_output/monochromatization_125](../../benchmarks/example_output/monochromatization_125).
Redraw the figures without running FeeBL:

```sh
python benchmarks/analysis/render_monochromatization.py \
  --data benchmarks/example_output/monochromatization_125/figure_data.json \
  --output results/monochromatization
```

## Smaller-energy and optics examples


These six cards illustrate horizontal dispersion with a simultaneous change
in the horizontal beta function. They cover ILC-like 250 GeV and Z-pole
collisions and an FCC-ee-like Z-pole collision. They are input examples,
without a claim of converged luminosity or peak-width accuracy. The 125 GeV
Higgs studies and their saved results are in [benchmarks](../../benchmarks/README.md)
and the [paper](../../docs/FeeBL_Manual_Paper.pdf).

Run from a POSIX shell, with an absolute path to the built executable:

```bash
FEEBL=/absolute/path/to/feebl/build/src/feebl ./run_all.sh
```

The script writes `out/<card-name>/` below this directory and copies each card
to `acc.dat` before execution. Repeating it overwrites files in those run
directories. Use a fresh copy of the examples to retain previous results.
The printed `#rms_W` is the RMS of the full spectrum, including radiative
tails; it is not a fitted peak width.

| Card | Nominal centre-of-mass energy | Horizontal beta function | Beam-1 dispersion |
|---|---:|---:|---:|
| `ilc250_baseline.dat` | 250 GeV | 13 mm | 0 |
| `ilc250_monochromatized.dat` | 250 GeV | 3 mm | 0.240258 mm |
| `ilcz_baseline.dat` | 91.2 GeV | 18 mm | 0 |
| `ilcz_monochromatized.dat` | 91.2 GeV | 4 mm | 0.328749 mm |
| `fccz_baseline.dat` | 91.2 GeV | 100 mm | 0 |
| `fccz_monochromatized.dat` | 91.2 GeV | 15 mm | 2.978622 mm |

Beam 2 has opposite dispersion in each nonzero-dispersion example. The ILC
cards use a 14 mrad full crossing angle and the ideal crab map. The FCC-ee
cards use a 30 mrad full crossing angle and the crab-waist map. These maps
describe distinct incoming geometries; a crab waist does not compensate the
longitudinal bunch tilt.

## Optics and settings

For fixed emittance and energy, reducing `beta_x` narrows the betatron beam
size. Dispersion adds a position-energy correlation and increases the total
horizontal size. In the independent Gaussian, zero-angle limit,

```text
sigma_beta = sigma_beta0 * sqrt(beta_x / beta_x0)
sigma_x,total^2 = sigma_beta^2 + (D_x * sigma_delta)^2.
```

Choosing the second term to restore the original size can offset a geometric
luminosity penalty in this limit. Finite crossing angle, hourglass effects,
disruption, unequal beam spreads, and radiation change the result. The cards
change both focusing and dispersion, so their difference does not isolate
dispersion alone. A final-focus lattice, chromaticity, apertures, and
synchrotron-radiation limits require separate studies.

- `ip_dispersion.1` and `.2` are in metres per unit fractional energy change.
  Opposite signs select compensating energy deviations. Same-sign dispersion
  provides a useful control, but is an exact null for the collision-energy
  selection only under the appropriate symmetric-beam assumptions.
- `do_espread=1` enables the incoming energy distribution required for this
  correlation. `which_espread` and `espread` specify its form and size.
- `spectrum_w_min`, `spectrum_w_max`, and `spectrum_w_bins` set the native
  luminosity-spectrum range and resolution. Resolve the peak before fitting
  it, and test binning and fit-window stability.
- The longitudinal sign assignments describe each beam's local coordinate
  convention and are set separately. Copy the complete geometry block when
  reproducing an example.

The ILC cards use 200,000 macroparticles per beam and the FCC-ee cards use
100,000. They use a single seed. Both absolute results and ratios require
independent-seed, domain, grid, slice, and macroparticle convergence checks.
See the [refinement plan](../../docs/FeeBL_Manual_Paper.pdf).
