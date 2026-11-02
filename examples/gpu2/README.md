# GPU mode 1 and GPU2 input-card fragments

These files are assignment fragments for an existing FeeBL `$PARAMETERS`
block; they are not complete accelerator cards.  Insert one fragment before a
known assignment such as `hist_ee_bins=...`, or pass the same text through the
`--deck-fragment` option of `tools/benchmark_backends.py`.  Do not concatenate
it after the closing `}` of a parameter block.

| Fragment | Purpose | Required observation |
|---|---|---|
| `mode1-atomic.fragment.dat` | CUDA CIC baseline | actual mode 1 and nonzero mode-1 GPU calls |
| `gpu2-fields-deterministic.fragment.dat` | One-host-thread, qualification-oriented fields scope | actual mode 2, `host_rho` fields plan, nonzero GPU2 calls, no fallback |
| `gpu2-fields-fast.fragment.dat` | Explicit fast-reduction fields ablation | actual `host_rho` fields plan; compare only after tolerance tests |
| `gpu2-fields-atomic-fast.fragment.dat` | Restricted direct resident CIC plus cuFFT candidate | actual `atomic` fields plan, nonzero deposition and solve calls, compact-metric/phi host boundary; CIC only, no coherent/trident field charge |
| `gpu2-fields-hybrid.fragment.dat` | Two-worker host CIC plus resident cuFFT candidate | OpenMP-enabled build, actual `gpu2_openmp_host`, two host threads, byte-identical deterministic output versus the one-thread `host_rho` control |
| `gpu2-primary-fast.fragment.dat` | Atomic resident CIC/cuFFT/device gather plus unchanged host primary physics | actual `primary`, compact-field boundary, host-serial push/BMT/ST/beamstrahlung and legacy RNG order, no host phi materialization |
| `gpu2-full-unsupported.fragment.dat` | Negative preflight test | nonzero exit until the full documented physics domain is implemented and qualified |
| `gpu2-cpu-fallback.fragment.dat` | Explicit CPU-only fallback test | provenance must report the fallback and zero GPU2 execution calls |

Omitting `gpu_mode` is intentionally not demonstrated for new cards: omission
retains mode 1 for backward compatibility. The full file remains a fail-fast
fixture, not a promise that full scope works. The fallback file is
for resilience testing only; a fallback run is invalid as a GPU timing sample.
The default fields scope uses `gpu2_deposition="auto"`, which resolves to the
`host_rho` boundary. Explicit `atomic` is also connected to
the whole-program path, but only for `electron_distribution_rho=2` with no
coherent or trident field-charge sources and `reduction_mode="fast"`.
`shared_tile` and `sort_reduce` remain fail-fast.  In particular,
deterministic atomic deposition is refused rather than mislabeled.

Primary scope is narrower than full residency. It requires the main grid only,
CIC deposition and FeeBL's internal fixed CIC interpolation, fast atomic
reduction, one stream and one host thread, graphs/fusion off, batch size one,
no coherent/trident field charge, no tracked pairs or muons, and `do_prod=0`.
`interpolation` remains an internal invariant, not an input-card key. It preserves primary
beamstrahlung, BMT, ST spin flip and all incoming geometry maps by feeding
compact device-gathered fields to the existing serial mover in beam order.

For a non-overwriting build/test evidence campaign, run:

```bash
bash tools/reproduce_gpu2.sh --output /new/path/feebl-gpu2-evidence
```

Benchmarks are disabled unless `--benchmarks` is supplied.  `--quick` produces
a labeled smoke artifact that cannot support performance claims.  Every
campaign requires separate powered stochastic, full physics-domain, sanitizer,
and native-Linux validation before supporting a production claim. In fields
scope, a combined CUDA+OpenMP build can set
`gpu2_host_threads=2`; exactly the two disjoint primary CIC depositions run as
OpenMP sections.  Luminosity, RNG, results, coherent/trident additions, and
all downstream physics remain serial. Measure any speedup with the full
workload and the same numerical settings. Primary scope does not use that
two-section host path: deposition is resident and `gpu2_host_threads>1` is
refused.
