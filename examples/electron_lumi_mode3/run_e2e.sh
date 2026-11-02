#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /absolute/path/to/feebl" >&2
  exit 2
fi

binary="$(readlink -f "$1")"
root="$(cd "$(dirname "$0")/../.." && pwd)"
deck="$root/examples/electron_lumi_mode3/tiny_heap.dat"
work="$(mktemp -d "${TMPDIR:-/tmp}/feebl-electron-lumi-mode3.XXXXXX")"
trap 'rm -rf -- "$work"' EXIT
export OMP_DYNAMIC=FALSE

reject () {
  local profile="$1" expected="$2" status
  set +e
  (
    cd "$work"
    "$binary" --acc_file="$deck" tiny_heap "$profile" "$profile.out" \
      >"$profile.stdout" 2>"$profile.stderr"
  )
  status=$?
  set -e
  if [[ $status -ne 2 ]]; then
    echo "$profile: expected fail-closed exit 2, observed $status" >&2
    exit 1
  fi
  grep -q "$expected" "$work/$profile.stderr"
  printf 'PASS: %-32s exit=2 diagnostic=%s\n' "$profile" "$expected"
}

reject rejected_invalid_mode "must be 0, 1, 2, or 3"
reject rejected_invalid_sampling "electron_lumi_sampling must be 0, 1, 2, or 3"
reject rejected_geometric_without_fullwave \
  "electron_lumi_sampling=3 requires wavefront_parallel_physics=1"
reject rejected_geometric_capacity "success-limit"
reject rejected_invalid_wave_physics "wavefront_parallel_physics must be 0 or 1"
reject rejected_wave_physics_contract "wavefront_parallel_physics=1 requires"
reject rejected_wave_physics_sink "shared output sinks"
reject rejected_sampling_without_mode3 "requires electron_lumi_precompute=3"
reject rejected_mode3_without_heap "do_lumi=1"
reject rejected_mode3_isr "do_isr=0"
reject rejected_mode3_coherent "do_coherent=0"
reject rejected_mode3_extra_grid "grids=1"

run_case () {
  local directory="$1" profile="$2"
  mkdir -p "$work/$directory"
  if ! (
    cd "$work/$directory"
    "$binary" --acc_file="$deck" tiny_heap "$profile" result.out \
      >stdout.txt 2>stderr.txt
  ); then
    echo "$profile: positive run failed; retained diagnostic follows" >&2
    sed -n '1,200p' "$work/$directory/stderr.txt" >&2
    exit 1
  fi
}

run_case legacy legacy_heap
run_case mode3_legacy mode3_legacy
run_case counter_a mode3_counter
run_case counter_b mode3_counter
run_case counter_serial mode3_counter_serial
run_case fullwave_2 fullwave_counter_2
run_case fullwave_4a fullwave_counter_4
run_case fullwave_4b fullwave_counter_4
run_case bernoulli_1 mode3_bernoulli_1
run_case bernoulli_4a mode3_bernoulli_4
run_case bernoulli_4b mode3_bernoulli_4
run_case fullwave_bernoulli_2 fullwave_bernoulli_2
run_case fullwave_bernoulli_4 fullwave_bernoulli_4
run_case fullwave_geometric_2 fullwave_geometric_2
run_case fullwave_geometric_4a fullwave_geometric_4
run_case fullwave_geometric_4b fullwave_geometric_4

cmp "$work/legacy/rndm.save" "$work/mode3_legacy/rndm.save"
cmp "$work/legacy/lumi.ee.out" "$work/mode3_legacy/lumi.ee.out"
cmp "$work/legacy/lumi.ee.out.fne" "$work/mode3_legacy/lumi.ee.out.fne"
cmp "$work/legacy/lumi.ee.out.dLdW.fne" \
    "$work/mode3_legacy/lumi.ee.out.dLdW.fne"

python3 - "$work/legacy/result.out" "$work/mode3_legacy/result.out" \
  "$work/counter_a/result.out" <<'PY'
import pathlib
import re
import sys

base, legacy, counter = map(pathlib.Path, sys.argv[1:])
marker = re.compile(br"^ electron_lumi_precompute = 3 .*\n", re.MULTILINE)
reference = base.read_bytes()
for label, path in (("mode3 legacy", legacy), ("mode3 counter", counter)):
    payload = path.read_bytes()
    matches = marker.findall(payload)
    if len(matches) != 1:
        raise SystemExit(f"{label}: expected exactly one mode3 provenance line")
    stripped = marker.sub(b"", payload)
    if stripped != reference:
        for line, pair in enumerate(
                zip(reference.splitlines(), stripped.splitlines()), 1):
            if pair[0] != pair[1]:
                raise SystemExit(
                    f"{label}: aggregate mismatch at line {line}: {pair!r}")
        raise SystemExit(f"{label}: aggregate output length mismatch")
print("PASS: both mode3 sampling policies preserve exact luminosity aggregates")
PY

for artifact in result.out lumi.ee.out lumi.ee.out.fne lumi.ee.out.dLdW.fne; do
  cmp "$work/counter_a/$artifact" "$work/counter_b/$artifact"
done
for artifact in lumi.ee.out lumi.ee.out.fne lumi.ee.out.dLdW.fne; do
  cmp "$work/counter_a/$artifact" "$work/counter_serial/$artifact"
done

grep -Eq 'parallel_evaluated_pairs=[1-9][0-9]*' \
  "$work/mode3_legacy/stderr.txt"
grep -Eq 'max_workers=4' "$work/mode3_legacy/stderr.txt"
grep -Eq 'parallel_sampled_candidates=0 .*legacy_heap_candidates=[1-9][0-9]*' \
  "$work/mode3_legacy/stderr.txt"
grep -Eq 'parallel_sampled_candidates=[1-9][0-9]* .*serial_sample_merge_candidates=[1-9][0-9]*' \
  "$work/counter_a/stderr.txt"
grep -Eq 'legacy_heap_candidates=0 .*counter_heap_accepts=[1-9][0-9]*' \
  "$work/counter_a/stderr.txt"
grep -q 'output_order=stable-pair-identity' "$work/counter_a/stderr.txt"
grep -Eq 'max_workers=1' "$work/counter_serial/stderr.txt"

awk 'NF { if ($NF != ++n) exit 1 } END { if (n != 100) exit 1 }' \
  "$work/counter_a/lumi.ee.out"

echo "PASS: mode3 legacy heap is byte-identical; counter sampling is repeatable and worker-count invariant"

for artifact in lumi.ee.out lumi.ee.out.fne lumi.ee.out.dLdW.fne; do
  cmp "$work/fullwave_2/$artifact" "$work/fullwave_4a/$artifact"
  cmp "$work/fullwave_4a/$artifact" "$work/fullwave_4b/$artifact"
done

python3 - "$work/fullwave_2/result.out" \
  "$work/fullwave_4a/result.out" "$work/fullwave_4b/result.out" <<'PY'
import pathlib
import sys

volatile = (
    "collision_schedule_requested =",
    "cpu_variant_requested =",
    "backend_provenance =",
)

def physics_lines(path):
    return [line for line in pathlib.Path(path).read_bytes().splitlines()
            if not any(line.startswith(prefix.encode()) for prefix in volatile)]

reference = physics_lines(sys.argv[1])
for candidate in sys.argv[2:]:
    if physics_lines(candidate) != reference:
        raise SystemExit(f"full-wave physics output mismatch: {candidate}")
print("PASS: full-wave physics outputs are exact across 2/4 workers and repeat")
PY

grep -q 'actual=wavefront_parallel_full_host_physics' \
  "$work/fullwave_2/stderr.txt"
grep -q 'full_parallel_make_step_encounters=36' \
  "$work/fullwave_4a/stderr.txt"
grep -q 'crossing_lumi_heap_publications=1' \
  "$work/fullwave_4a/stderr.txt"
grep -Eq \
  'collision_schedule_actual = wavefront_parallel_full_host_physics scheduler = encounter-dag-worker-private-host-physics-v1 .*ordered_commit = 0 .*physics_equivalence_scope = fccz-primary-beamstrahlung-photon-full-ee-luminosity pair-aggregate-luminosity = full-ee' \
  "$work/fullwave_4a/result.out"
test "$(wc -l < "$work/fullwave_4a/lumi.ee.out")" -eq 101

echo "PASS: full-wave worker-local collisions publish one exact crossing reservoir"

for artifact in lumi.ee.out lumi.ee.out.fne lumi.ee.out.dLdW.fne; do
  cmp "$work/bernoulli_1/$artifact" "$work/bernoulli_4a/$artifact"
  cmp "$work/bernoulli_4a/$artifact" "$work/bernoulli_4b/$artifact"
done

python3 - "$work/bernoulli_1/result.out" \
  "$work/bernoulli_4a/result.out" "$work/bernoulli_4b/result.out" <<'PY'
import pathlib
import sys

volatile = (
    " electron_lumi_precompute =",
    "compute_backend_requested =",
    "cpu_variant_requested =",
    "gpu_variant_requested =",
    "gpu_mode_requested =",
    "gpu2_scope =",
    "backend_provenance =",
)

def physics_lines(path):
    return [line for line in pathlib.Path(path).read_bytes().splitlines()
            if not any(line.startswith(prefix.encode()) for prefix in volatile)]

reference = physics_lines(sys.argv[1])
for candidate in sys.argv[2:]:
    if physics_lines(candidate) != reference:
        raise SystemExit(f"Bernoulli aggregate output mismatch: {candidate}")
print("PASS: keyed Bernoulli aggregates and events are exact across 1/4 workers")
PY

grep -Eq 'sampling_mode=2 .*bernoulli_draws=[1-9][0-9]* .*bernoulli_rejects=[1-9][0-9]* .*bernoulli_accepted_pairs=[1-9][0-9]*' \
  "$work/bernoulli_4a/stderr.txt"
records="$(awk 'NF { ++n; if ($NF != n) exit 1 } END { print n+0 }' \
  "$work/bernoulli_4a/lumi.ee.out")"
test "$records" -gt 0
test "$records" -lt 100

for artifact in lumi.ee.out lumi.ee.out.fne lumi.ee.out.dLdW.fne; do
  cmp "$work/fullwave_bernoulli_2/$artifact" \
      "$work/fullwave_bernoulli_4/$artifact"
done
grep -q 'sampling_mode=2' "$work/fullwave_bernoulli_4/stderr.txt"
grep -q 'crossing_lumi_heap_publications=1' \
  "$work/fullwave_bernoulli_4/stderr.txt"

python3 - \
  "$work/fullwave_bernoulli_2/stderr.txt" \
  "$work/fullwave_bernoulli_4/stderr.txt" <<'PY'
import pathlib
import re
import sys

summary_prefix = "FEEBL electron-lumi summary:"

def summaries(path):
    lines = [line for line in pathlib.Path(path).read_text().splitlines()
             if line.startswith(summary_prefix)]
    if not lines:
        raise SystemExit(f"no electron-luminosity telemetry in {path}")
    result = []
    for line in lines:
        fields = {key: int(value) for key, value in
                  re.findall(r"([a-z0-9_]+)=([0-9]+)", line)}
        for key in ("bernoulli_draws", "bernoulli_rejects",
                    "bernoulli_accepted_pairs", "bernoulli_accepted_records",
                    "counter_heap_accepts"):
            if key not in fields:
                raise SystemExit(f"missing {key} telemetry in {path}")
        result.append(fields)
    return result

def total(rows, field):
    return sum(row[field] for row in rows)

mode2_2, mode2_4 = map(summaries, sys.argv[1:])
# Modes 1 and 2 intentionally retain their measured non-fused worker-local
# paths. Mode-1 reservoir-operation telemetry is partition-dependent; its
# crossing-level canonical top-K output is checked byte-for-byte above. Mode 2
# is append-only, so its accepted/draw/reject totals remain worker-invariant.
for field in ("bernoulli_draws", "bernoulli_rejects",
              "bernoulli_accepted_pairs", "bernoulli_accepted_records",
              "counter_heap_accepts"):
    if total(mode2_2, field) != total(mode2_4, field):
        raise SystemExit(f"mode2: 2/4-worker {field} totals differ")
for rows in (mode2_2, mode2_4):
    if total(rows, "bernoulli_draws") != \
            total(rows, "bernoulli_rejects") + \
            total(rows, "bernoulli_accepted_pairs"):
        raise SystemExit("mode2: Bernoulli draw accounting does not close")
    if total(rows, "bernoulli_accepted_records") != \
            total(rows, "counter_heap_accepts"):
        raise SystemExit("mode2: append-only record accounting does not close")

print("PASS: append-only mode-2 counters are 2/4-worker invariant")
PY

echo "PASS: mode 2 remains sparse, repeatable, and full-wave worker invariant"

for artifact in lumi.ee.out lumi.ee.out.fne lumi.ee.out.dLdW.fne; do
  cmp "$work/fullwave_geometric_2/$artifact" \
      "$work/fullwave_geometric_4a/$artifact"
  cmp "$work/fullwave_geometric_4a/$artifact" \
      "$work/fullwave_geometric_4b/$artifact"
done

python3 - "$work/fullwave_geometric_2/result.out" \
  "$work/fullwave_geometric_4a/result.out" \
  "$work/fullwave_geometric_4b/result.out" <<'PY'
import pathlib
import sys

volatile = (
    "collision_schedule_requested =",
    "cpu_variant_requested =",
    "backend_provenance =",
)

def physics_lines(path):
    return [line for line in pathlib.Path(path).read_bytes().splitlines()
            if not any(line.startswith(prefix.encode()) for prefix in volatile)]

reference = physics_lines(sys.argv[1])
for candidate in sys.argv[2:]:
    if physics_lines(candidate) != reference:
        raise SystemExit(f"geometric full-wave physics output mismatch: {candidate}")
print("PASS: geometric full-wave artifacts are exact across 2/4 workers and repeat")
PY

python3 - \
  "$work/fullwave_geometric_2/stderr.txt" \
  "$work/fullwave_geometric_4a/stderr.txt" \
  "$work/fullwave_geometric_4b/stderr.txt" \
  "$work/fullwave_geometric_2/lumi.ee.out" <<'PY'
import pathlib
import re
import sys

summary_prefix = "FEEBL electron-lumi summary:"
required = (
    "logical_pairs", "parallel_evaluated_pairs",
    "parallel_sampled_candidates", "serial_sample_merge_candidates",
    "counter_heap_accepts", "sampling_mode", "bernoulli_draws",
    "bernoulli_rejects", "bernoulli_accepted_pairs",
    "bernoulli_accepted_records", "outer_fused_pairs", "geometric_rows",
    "geometric_rng_calls", "geometric_successes",
    "geometric_integer_copies", "geometric_fallback_rows",
    "geometric_fallback_pairs",
)

def summaries(path):
    lines = [line for line in pathlib.Path(path).read_text().splitlines()
             if line.startswith(summary_prefix)]
    if not lines:
        raise SystemExit(f"no electron-luminosity telemetry in {path}")
    rows = []
    for line in lines:
        fields = {key: int(value) for key, value in
                  re.findall(r"([a-z0-9_]+)=([0-9]+)", line)}
        missing = [field for field in required if field not in fields]
        if missing:
            raise SystemExit(f"missing telemetry {missing} in {path}")
        if fields["sampling_mode"] != 3:
            raise SystemExit(f"non-geometric sampling summary in {path}")
        if fields["serial_sample_merge_candidates"] != 0:
            raise SystemExit(f"serial sample merge ran in {path}")
        if fields["geometric_fallback_rows"] != 0 or \
                fields["geometric_fallback_pairs"] != 0:
            raise SystemExit(f"uniform stable-ID card used fallback in {path}")
        if fields["logical_pairs"] != fields["outer_fused_pairs"] or \
                fields["parallel_evaluated_pairs"] != \
                fields["outer_fused_pairs"] or \
                fields["parallel_sampled_candidates"] != \
                fields["outer_fused_pairs"]:
            raise SystemExit(f"not every logical pair used fused mode 3 in {path}")
        rows.append(fields)
    return rows

def totals(rows):
    return {field: sum(row[field] for row in rows) for field in required}

runs = [totals(summaries(path)) for path in sys.argv[1:4]]
stable_fields = tuple(field for field in required if field != "sampling_mode")
for field in stable_fields:
    values = [run[field] for run in runs]
    if len(set(values)) != 1:
        raise SystemExit(f"geometric 2/4/repeat {field} totals differ: {values}")

stats = runs[0]
if stats["geometric_rows"] <= 0:
    raise SystemExit("geometric cursor did not process any rows")
if not 0 < stats["geometric_rng_calls"] < stats["bernoulli_draws"]:
    raise SystemExit("geometric RNG calls did not beat logical opportunities")
if stats["bernoulli_draws"] != stats["logical_pairs"]:
    raise SystemExit("logical Bernoulli opportunity accounting does not close")
if stats["geometric_integer_copies"] != 0:
    raise SystemExit("tiny store<1 card unexpectedly emitted integer copies")
if stats["bernoulli_draws"] != \
        stats["bernoulli_rejects"] + stats["geometric_successes"]:
    raise SystemExit("geometric Bernoulli draw accounting does not close")
if stats["bernoulli_accepted_pairs"] != stats["geometric_successes"]:
    raise SystemExit("geometric accepted-pair accounting does not close")
if stats["bernoulli_accepted_records"] != \
        stats["geometric_successes"] + stats["geometric_integer_copies"]:
    raise SystemExit("geometric accepted-record accounting does not close")
if stats["counter_heap_accepts"] != stats["bernoulli_accepted_records"]:
    raise SystemExit("append-only heap accounting does not close")

records = 0
for line in pathlib.Path(sys.argv[4]).read_text().splitlines():
    if line.strip():
        records += 1
        if int(line.split()[-1]) != records:
            raise SystemExit("geometric native output order is not contiguous")
if records != stats["bernoulli_accepted_records"]:
    raise SystemExit(
        f"geometric output/telemetry record mismatch: {records} != "
        f"{stats['bernoulli_accepted_records']}")

print("PASS: geometric fusion, skip efficiency, and accounting are exact")
PY

echo "PASS: mode 3 geometric sampling is fused, sparse, and worker invariant"
