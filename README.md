# FeeBL: Fast $e^{\pm}e^{\pm}$ Beam and Luminosity

FeeBL simulates an electron-positron bunch crossing, including luminosity
spectra, beamstrahlung, pair production, beam deflection and spin transport.
It extends GUINEA-PIG++ 1.2.2. See [attribution](ORIGIN.md) and the supplied
[distribution terms](ORIGIN.md#distribution-terms).

## Build

The CPU build needs CMake 3.16 or later and a C++17 compiler. FFTW3 is
recommended. On Ubuntu or WSL2:

```sh
sudo apt-get install build-essential cmake libfftw3-dev libhdf5-dev python3 python3-venv nodejs
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DFEEBL_ENABLE_CUDA=OFF
cmake --build build --parallel 4
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-analysis.txt
```

The executable is `build/src/feebl`. Native Windows and macOS builds are not
qualified. Python 3.10 or later runs the analysis tools; the browser tests
require Node.js 18 or later. For CUDA builds, set `FEEBL_ENABLE_CUDA=ON` and
`FEEBL_CUDA_ARCHITECTURES` to the device architecture. See the
[manual](docs/FeeBL_Manual_Paper.pdf) for build options and supported combinations.

## Run a collision

```sh
python benchmarks/run.py verify
python benchmarks/run.py run smoke --feebl build/src/feebl --output runs/smoke
```

The smoke example checks installation with a small fixed mesh. The runner
uses a fresh output directory and records the input and executable hashes.
For a custom card, place `acc.dat` in a new working directory and run:

```sh
/absolute/path/to/feebl accelerator_block parameter_block out.dat > log.txt 2>&1
```

Block names are declared in the card. The [input reference](docs/INPUT_REFERENCE.md)
lists keywords, types, units and defaults. [Coordinate conventions in the manual](docs/FeeBL_Manual_Paper.pdf)
and the [output reference](OUTPUT_FILES) describe stored quantities.

## Beam viewer

```sh
python tools/serve_viewer.py --run benchmarks/example_output/gui_smoke
```

The server prints a local browser URL. The viewer provides frame playback,
luminosity and particle spectra, polarization, beam moments, and field layers
when those quantities were recorded. See the [manual](docs/FeeBL_Manual_Paper.pdf)
for finalizing new runs, selections and exports.

## Monochromatization examples

The [125 GeV scan guide](examples/monochromatization/README.md) describes the
40 additional ILC/LCF, HALHF and FCC-ee cards, CPU/GPU requirements and plot
reproduction. Cards are registered with `benchmarks/run.py list`.

```sh
python benchmarks/run.py run fcc_cw_d175_highres_s1 --feebl build/src/feebl \
  --output runs/fcc175 --timeout 28800 --memory-gib 0
python benchmarks/analysis/render_monochromatization.py \
  --data benchmarks/example_output/monochromatization_125/figure_data.json \
  --output results/monochromatization
```

These large cards require suitable compute resources; the smoke example is
the installation test. The study varies horizontal dispersion. Other
monochromatization mechanisms and storage-ring equilibrium are outside its scope.

## Tests, benchmarks and paper

```sh
ctest --test-dir build -LE strict --output-on-failure -j 1
ctest --test-dir build -L strict --output-on-failure -j 1
python tools/check_browser_modules.py
```

The strict adaptive-field test is a separate numerical qualification and
currently fails its accuracy requirements. Excluding it runs software checks
only. See the [test guide](examples/README.md), [benchmark guide](benchmarks/README.md)
and [analysis recipes](benchmarks/analysis/README.md).

The [manual and benchmark paper](docs/FeeBL_Manual_Paper.pdf) is included as
a PDF. The latest 125 GeV scan figures and updated tables are in
`benchmarks/example_output/monochromatization_125`. LaTeX sources are supplied
separately.

Full historical particle records are supplied separately in
`FeeBL_Reproduction_Data_20261030.zip`. They are not needed to build FeeBL,
run a new collision or redraw the compact monochromatization figures.
To restore them, use `python tools/reproduction_data.py restore ARCHIVE --dry-run`
first, then repeat without `--dry-run`; `python tools/reproduction_data.py verify`
checks the restored files.

`examples/` contains input examples, regression tests and their CMake
registration. `tools/` contains the viewer, data conversion and analysis
utilities. `benchmarks/` contains registered cases, reference data and compact
results. The [input reference](docs/INPUT_REFERENCE.md) lists all native keys.

## AI Disclosure

This work was done with a combination of agentic AI and human labor. No generative AI was used in any part.
