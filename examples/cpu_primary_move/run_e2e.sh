#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 /absolute/path/to/feebl [evidence-directory]" >&2
  exit 2
fi

binary="$(readlink -f "$1")"
evidence="${2:-}"
if [[ -n "$evidence" ]]; then
  if [[ -e "$evidence" ]]; then
    echo "evidence directory already exists: $evidence" >&2
    exit 2
  fi
  evidence="$(readlink -m "$evidence")"
fi
root="$(cd "$(dirname "$0")/../.." && pwd)"
template="$root/examples/cpu_primary_move/tiny.dat"
work="$(mktemp -d "${TMPDIR:-/tmp}/feebl-cpu-primary-move.XXXXXX")"
trap 'rm -rf -- "$work"' EXIT

make_deck()
{
  local destination="$1" replacement="$2"
  mkdir -p "$destination"
  sed "s|cpu_primary_move=0;|$replacement|" "$template" \
    >"$destination/acc.dat"
}

run_case()
{
  local name="$1" replacement="$2"
  make_deck "$work/$name" "$replacement"
  (
    cd "$work/$name"
    "$binary" tiny exact result.out >stdout.txt 2>stderr.txt
  )
}

normalize_result()
{
  grep -vE \
    ' cpu_primary_move = |compute_backend_requested|cpu_variant_requested|gpu_variant_requested|gpu_mode_requested|gpu2_|backend_fallback|backend_provenance' \
    "$1"
}

compare_exact()
{
  local reference="$1" candidate="$2"
  cmp <(normalize_result "$reference/result.out") \
      <(normalize_result "$candidate/result.out")
  cmp "$reference/beam1.dat" "$candidate/beam1.dat"
  cmp "$reference/beam2.dat" "$candidate/beam2.dat"
  cmp "$reference/rndm.save" "$candidate/rndm.save"
}

reject()
{
  local name="$1" replacement="$2" expected="$3" status=0
  make_deck "$work/$name" "$replacement"
  (
    cd "$work/$name"
    "$binary" tiny exact result.out >stdout.txt 2>stderr.txt
  ) || status=$?
  if [[ $status -eq 0 ]]; then
    echo "$name: unsupported cpu_primary_move card ran instead of failing" >&2
    exit 1
  fi
  grep -q "$expected" "$work/$name/stderr.txt"
}

# The omitted/default selector must leave the historical serial route silent.
run_case oracle ''
if grep -q 'cpu_primary_move' "$work/oracle/result.out" ||
   grep -q 'cpu_primary_move' "$work/oracle/stderr.txt"; then
  echo "default-off run unexpectedly emitted primary-movement provenance" >&2
  exit 1
fi

for threads in 1 2 4; do
  run_case "cpu$threads" \
    "compute_backend=\"cpu\";cpu_threads=$threads;cpu_variant=\"pool_dual\";reduction_mode=\"strict\";cpu_primary_move=1;"
  compare_exact "$work/oracle" "$work/cpu$threads"
  grep -Eq \
    'FEEBL: final .*cpu_primary_move=pool-static-primary-v1.*cpu_primary_move_calls=[1-9][0-9]*.*cpu_primary_move_particles=[1-9][0-9]*' \
    "$work/cpu$threads/stderr.txt"
done

# Every unsafe or ambiguous request is refused before the first collision.
reject invalid_selector 'cpu_primary_move=2;' \
  'cpu_primary_move must be 0 or 1'
reject serial_backend 'cpu_primary_move=1;' \
  'requires an active cpu backend'
reject beamstrahlung 'do_eloss=1;cpu_primary_move=1;' 'do_eloss=0'
reject spin_flip 'ST_spin_flip=1;cpu_primary_move=1;' 'ST_spin_flip=0'
reject bmt 'bmt_precession=1;cpu_primary_move=1;' 'bmt_precession=0'
reject trident 'do_trident=1;cpu_primary_move=1;' 'do_trident=0'
reject coherent 'do_coherent=1;cpu_primary_move=1;' 'do_coherent=0'
reject production 'do_prod=1;cpu_primary_move=1;' 'do_prod=0'
reject extra_grid 'grids=2;cpu_primary_move=1;' 'grids=1'

if [[ -n "$evidence" ]]; then
  mkdir -p "$evidence"
  cp -a "$work/." "$evidence/"
  sha256sum "$binary" "$template" >"$evidence/input-sha256.txt"
  {
    printf 'binary=%s\n' "$binary"
    printf 'source_root=%s\n' "$root"
    git -C "$root" rev-parse HEAD
    git -C "$root" status --short --branch
    uname -a
  } >"$evidence/provenance.txt"
fi

echo "PASS: default path silent; 1/2/4-worker primary movement is byte-exact"
echo "PASS: result, ordered beam records, and legacy RNG state are identical"
echo "PASS: unsupported physics and backend scopes fail closed"
