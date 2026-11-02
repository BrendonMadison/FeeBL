# Adaptive-grid example cards

These cards exercise stable input contracts. They are intentionally small
smoke cases; their outputs are not collider-physics or performance
qualification results.

Run from WSL2 with a clean Release/OpenMP build. Each invocation must use a
fresh directory because FeeBL reads `acc.dat` from its current directory and
writes fixed output names there:

```bash
set -euo pipefail
SOURCE=/absolute/path/to/feebl
CPU_BUILD="$SOURCE/build-adaptive-cpu"
GPU_BUILD="$SOURCE/build-adaptive-gpu"
EXAMPLES="$SOURCE/examples/adaptive_grid"
RUN_ROOT=/tmp/feebl-adaptive-examples-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$RUN_ROOT"

run_card() {
  name=$1
  binary=$2
  card=$3
  mkdir "$RUN_ROOT/$name"
  cp "$card" "$RUN_ROOT/$name/acc.dat"
  (cd "$RUN_ROOT/$name" &&
    "$binary" fccz p_fccz out.dat >log.txt 2>&1)
}

run_card mode0 "$CPU_BUILD/src/feebl" "$EXAMPLES/mode0_reference.dat"
run_card mode1_cpu "$CPU_BUILD/src/feebl" \
  "$EXAMPLES/mode1_cpu_exact_window.dat"
run_card mode1_gpu5 "$GPU_BUILD/src/feebl" \
  "$EXAMPLES/gpu5_mode1_exact_window.dat"
run_card mode2_cpu "$CPU_BUILD/src/feebl" \
  "$EXAMPLES/mode2_error_controlled_cpu.dat"
run_card mode3_cpu "$CPU_BUILD/src/feebl" \
  "$EXAMPLES/mode3_nested_patch_cpu.dat"
```

The CPU Mode-1 log must identify
`adaptive_grid_requested=1 adaptive_grid_actual=1`,
`requested=wavefront_parallel actual=wavefront_parallel_full_host_physics`,
and a saved schedule. The GPU log
must additionally contain `FEEBL_GPU5_COMPATIBILITY_V1`,
`requested_gpu_mode=5 actual_gpu_mode=5`, the `global-atomic` deposition
variant, and a saved schedule. Any fallback text, missing CUDA executor, or
nonzero exit is a failure. Compare the Mode-0 and CPU Mode-1 results only after
checking that both runs used the intended card and binary hashes.

Modes 2 and 3 are executable, experimental encounter-scope CPU modes. Their
cards explicitly request `compute_backend="cpu"`; an automatic/GPU request,
backend drift, or fallback is an error. Mode 2 solves the same pre-kick
particles on at least three consecutive ratio-two bootstrap meshes whose
physical spacings are `dx/f,dy/f`.  The card list is not a finest-spacing
ceiling: with `adaptive_grid_max_refinement_factor=0`, encounters append
ratio-two classes lazily until acceptance or a declared resource or host
representation limit binds. Update interval, Richardson interval, and shrink
dwell are all exactly one. It selects a level only when RMS and `L_inf`
Richardson, indicator, and weighted-sampling gates all permit it. Positive
measured contraction is required for every error-relevant component; a
component already below target under the conservative `p=0.25` bound is
exempt. Mode 3 conditionally proposes one connected ratio-two fine patch using
`adaptive_grid_coarsen_threshold` in `[0,1]` as its active normalized
overlap/charge cut. If maximum overlap is at or below
`adaptive_grid_rho_floor`, component selection falls back to maximum charge.
Its fine free-space particular solution receives a harmonic Dirichlet
correction from the coarse potential, and a smooth potential-level transition
protects the interface. Once a patch is requested, the encounter must record
either a certified patch or explicit promotion to the next
resource-admissible uniform-fine audit class; no-request, terminal-uniform,
and empty encounters are distinct valid states.

An additional default-off experimental route,
`adaptive_grid_terminal_patch_certification=1`, may be used only with the
continuous fixed-width source and the documented FD4/vector-quantile controls.
It attempts two same-support local levels beyond an unqualified terminal
uniform mesh and accepts only after the full composite proof closes.  The
stable examples leave it disabled.  See
`docs/ADAPTIVE_GRID_TERMINAL_PATCH_CERTIFICATION.md`; enabling it is not a
physics-qualification claim.

The example Mode-2/3 cards select `fd4_lagrange4_v1`.  Its interior gather is
central FD4 plus tensor four-node cubic interpolation.  On true global faces,
the interpolation nodes shift/clamp in-domain and the first/last two nodal
derivatives use five-point fourth-order one-sided closures.  Cropped local
faces remain fail-closed unless every required global support node is present.

The positive, even-width transition multiplies four independent
cubic-smoothstep ramps measured from physical faces at local coordinates `0`
and `n`. Cell-centre weights at `0.5` and `n-0.5` are mirror-symmetric in both
axes, and the four-ramp product remains globally C1 at overlapping corner
transitions. A patch rejected by weighted sampling skips all later patch-only
work; one rejected by the harmonic residual preserves that diagnostic but
skips composite/gather/oracle work. Both requested-patch rejection routes may
publish only through a qualified uniform-fine promotion.

Canonical encounter and stochastic identities remain stable, while the
density-overlap luminosity uses the selected physical cell area. The modes do
not combine a fine-grid density dot product with canonical-grid area
normalization.

The two logs must contain, respectively:

```bash
grep -F "adaptive grid summary: requested=2 actual=2" \
  "$RUN_ROOT/mode2_cpu/log.txt"
grep -E "adaptive spacing class: factor=[24] selected_encounters=[1-9]" \
  "$RUN_ROOT/mode2_cpu/log.txt"
grep -F "mode2_unmet_error_encounters=0" \
  "$RUN_ROOT/mode2_cpu/log.txt"

grep -F "adaptive grid summary: requested=3 actual=3" \
  "$RUN_ROOT/mode3_cpu/log.txt"
grep -E "mode3_(patch_encounters|uniform_promotions)=[1-9]" \
  "$RUN_ROOT/mode3_cpu/log.txt"
grep -F "mode2_unmet_error_encounters=0" \
  "$RUN_ROOT/mode3_cpu/log.txt"
grep -E "uniform_context_misses=[0-9]+ patch_context_misses=[0-9]+ context_evictions=[0-9]+" \
  "$RUN_ROOT/mode3_cpu/log.txt"
grep -E "mode3_no_patch_encounters=[0-9]+ mode3_empty_encounters=[0-9]+" \
  "$RUN_ROOT/mode3_cpu/log.txt"
grep -E "mode3_controller_inclusive_worker_ms=[0-9.eE+-]+" \
  "$RUN_ROOT/mode3_cpu/log.txt"
grep -E "cold_miss_count=[0-9]+" \
  "$RUN_ROOT/mode3_cpu/log.txt"
```

The grep commands check only that the intended route and a recorded Mode-3
outcome were reached. They do not establish that patch certification,
luminosity normalization, or physics accuracy is correct.

For lazy Modes 2/3, `cold_miss_count` is real work: it equals uniform plus
patch context misses and need not be zero. `context_evictions` records bounded
replacement, not an error. Lazy and transient contexts use deterministic
`FFTW_ESTIMATE`. With FFTW3 the harmonic correction is a direct
two-dimensional DST-I solve; without FFTW3 the compatibility path is
deterministic red-black SOR. The card's `adaptive_grid_max_bytes` must cover
both the retained context pool and a separately reserved conservative
transient high-water allowance per admitted lane; it is not a process-RSS
ceiling.

Immediately before collision consumption, adaptive prefetches recheck the
complete ordered particle-pointer and binary32 `(x,y)` snapshot plus all four
source/target descriptor counts. A stale prefetch must terminate instead of
publishing physics. `mode3_controller_inclusive_worker_ms` overlaps component
timers and must not be summed with them or with wall time.

Run `examples/adaptive_grid/run_modes23_physics_validation.sh` for the small
32^2/64^2/128^2 mechanism comparison. Interpret its output using the
patch-or-promotion and resolution-consistent-luminosity contracts above.
A successful run is an implementation regression, not 1--3% collider
qualification. The latter needs analytic
Gaussian/Bassetti--Erskine field tests, Gauss/self-force/interface tests,
paired multi-seed and `n_m` ladders, and fixed-grid ladders to asymptotic
closure.

GPU mode 5 remains Mode-1-only and refuses Modes 2/3 before beam generation;
schedule format 1 likewise refuses their save/load requests. Modes 2/3 do not
silently drift to the CPU from another requested backend. A future GPU
implementation needs spacing-aware cache keys, resolution-consistent
luminosity ownership, uniform-promotion support, and nested-patch CUDA kernels.

The exact grammar and current qualification boundary are documented in
`docs/ADAPTIVE_GRID_MODES23_CARD_CONTRACT.md`.

The optional mixed absolute/relative field-error policy must carry a
prospectively frozen budget binding. A card fragment is:

```text
adaptive_grid_error_norm_policy="vector_mixed_rms_quantile_v2";
adaptive_grid_error_budget_sha256="0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
adaptive_grid_error_reference_field=12.5;
adaptive_grid_error_absolute_floor_fraction=0.001;
adaptive_grid_error_catastrophe_absolute_fraction=0.01;
adaptive_grid_error_exposure_limit=0.001;
```

The shown digest and numeric values are grammar-only placeholders. Replace
them from one immutable error-budget document defined before the study, and
verify that document's bytes before
running. FeeBL rejects a missing, uppercase, short, or noncanonical digest and
rejects a nonempty digest under either legacy error policy.

The three `timing_*.dat` files preserve the larger cards used by the recorded
15-pair Mode-0 and five-pair CPU fixed/crop protocols. They are retained inputs,
not a claim that a new timing run was performed.
