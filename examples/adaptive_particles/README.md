# Adaptive-macroparticle example cards

These are small CPU tests of the projected field-source representation.
They are **not** FCC-ee convergence cards, performance benchmarks,
or evidence of 1% physics accuracy.

| Card | Adaptive grid | Adaptive particles | Expected source action |
|---|---:|---:|---|
| `diagnostics_cpu_grid2.dat` | 2 | 1 | Build and verify an uncommitted proposal, retain it for encounter-local audit, then discard it with that hierarchy. |
| `mode2_split_cpu_grid2.dat` | 2 | 2 | Commit the exact nonzero CIC cell atoms. |
| `mode3_split_merge_cpu_grid3.dat` | 3 | 3 | Commit exact CIC atoms after same-cell/cohort merge. |
| `gpu6_unsupported_fail_closed.dat` | 2 | 2 | Negative capability example: request the reserved GPU Mode 6 and verify rejection before simulation. |

Every card uses the only currently admitted combination:

- explicit CPU `pool_dual`, strict reduction, no backend fallback;
- adaptive-grid Mode 2 or 3 with `reference_cic_v1`;
- `field_source + projected_current_state`;
- one statistical cohort and the fixed v1 controller schedule;
- no policy/trace persistence.

The two active mechanism cards set
`adaptive_macroparticle_fail_on_unmet_error=0`. This is intentional and does
not weaken or pass a physics gate: their single complete realization cannot
estimate initial-source sampling uncertainty, so the controller reports its
total bound as unavailable and the final qualification remains software-only.
Changing that flag to its production default `1` is a negative test: the run
must stop before writing final summary physics files and report that
independent-run evidence is required. FeeBL emits a versioned replay signal
but does not invoke the driver or insert particles during the collision.
The standalone ordinary-MC workflow under `examples/adaptive_particles/`
must be planned and run separately. The signal and exploratory-output
semantics are described in [the replay driver guide](../../docs/FeeBL_Manual_Paper.pdf).

Run each card in a fresh directory because FeeBL reads `acc.dat` and writes
fixed output names in its current working directory:

```bash
set -euo pipefail
FEEBL=/absolute/path/to/feebl
SOURCE=/absolute/path/to/feebl/source
RUN_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/feebl-adaptive-particles.XXXXXXXX")

for card in \
  diagnostics_cpu_grid2 \
  mode2_split_cpu_grid2 \
  mode3_split_merge_cpu_grid3
do
  mkdir "$RUN_ROOT/$card"
  cp "$SOURCE/examples/adaptive_particles/$card.dat" \
     "$RUN_ROOT/$card/acc.dat"
  (cd "$RUN_ROOT/$card" &&
    "$FEEBL" fccz p_fccz out.dat >log.txt 2>&1)
done
```

Each active run must emit
`FEEBL_ADAPTIVE_MACROPARTICLE_PREFLIGHT_V1`. Mode 1 must report a populated
diagnostic proposal with zero commits and an unchanged live revision; its
representatives can never be deposited. Modes 2 and 3 must report bound source hashes,
population revisions, root and representative counts, conservation residuals,
and committed source status. Treat a missing receipt, backend fallback,
uncommitted Mode-2/3 source, stale binding, or nonzero process status as a
failure.

For a broader parser/preflight test, use
`examples/adaptive_particles/run_card_contract_validation.sh`. The complete
grammar and the qualification caveat are in
`docs/ADAPTIVE_MACROPARTICLE_CARD_CONTRACT.md`. The final integrated suite
reports 36 checks, including the GPU Mode 6 refusal below; this count is a
software-contract result, not physics qualification.

## GPU Mode 6 negative example

`gpu6_unsupported_fail_closed.dat` is intentionally **not** a runnable GPU
physics card. GPU Mode 6 is reserved for a future device-resident coupled
adaptive-grid/adaptive-particle implementation. That
implementation, its CUDA conservation reductions, and its collider
qualification are absent. The card therefore supplies a
reproducible negative contract: the current executable must exit nonzero and
must not publish `out.dat` or luminosity files.  Never enable backend fallback
when testing this boundary, and never relabel an existing GPU mode as Mode 6.
